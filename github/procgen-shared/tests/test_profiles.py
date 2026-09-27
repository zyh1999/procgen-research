"""Dense, streaming and entry-point checks; no Procgen training is launched."""
import copy
import unittest
import torch
from test_algebra import TinyShared, score_fixture
from shared_procgen.algebra import LayerwiseIO
from shared_procgen.coefficient128 import solve_coeff128
from shared_procgen.streamed_coefficient128 import solve_streamed_coeff128
from shared_procgen.streamed_dual import solve_streamed
from shared_procgen.subcurvature import solve_dual_coefficients_exactfusion
from shared_procgen.update import update
from shared_procgen.large_update import update_large
from shared_procgen.training import PROFILES, build_parser


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixedtail_ray_against_dense_sample_system(self):
        _, factors, h = score_fixture(batch=160)
        anchors = torch.arange(128)
        targets = torch.stack((torch.randn(160), torch.ones(160)), 1)
        ratios = torch.stack((torch.linspace(.1, 3, 160), torch.ones(160)), 1)
        actual, info = solve_coeff128(factors, targets, ratios, anchors, .5, 160)
        for j in range(2):
            root = ratios[:, j].double().sqrt()
            system = root[:, None]*(h@h.T/160)*root[None, :] + .5*torch.eye(160).double()
            rhs = root*targets[:, j].double()
            x = rhs/.5
            x[:128] = torch.linalg.solve(system[:128, :128], rhs[:128]-system[:128, 128:]@x[128:])
            response = system@x
            gamma = rhs@response/(response@response)
            expected = gamma*x/root
            torch.testing.assert_close(actual[:, j].double(), expected, atol=3e-5, rtol=3e-5)
            self.assertLessEqual(info[j]['sample_residual2_after'], info[j]['sample_residual2_before']+1e-8)
            self.assertEqual(info[j]['cholesky_info'], 0)

    def test_streaming_both_solvers_matches_full_factors_and_chunks(self):
        torch.manual_seed(31)
        model = TinyShared()
        batch = 160
        obs, noise = torch.randn(batch, 4), torch.randn(batch)
        actions = torch.arange(batch)%3
        targets = torch.stack((torch.randn(batch), torch.ones(batch)), 1)
        ratios = torch.stack((torch.linspace(.1, 3, batch), torch.ones(batch)), 1)
        with LayerwiseIO(model) as io:
            values, logits = model(obs)
        factors = io.factors(logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)+2*noise*values)
        for method in ('dual127p1', 'coeff128'):
            anchors = torch.arange(127 if method == 'dual127p1' else 128)
            reference = solve_dual_coefficients_exactfusion if method == 'dual127p1' else solve_coeff128
            expected, _ = reference(factors, targets, ratios, anchors, .5, batch)
            for chunk in (37, 128):
                if method == 'dual127p1':
                    actual, _ = solve_streamed(model, obs, actions, targets, ratios, noise, anchors, True, batch, chunk=chunk)
                else:
                    actual, info = solve_streamed_coeff128(model, obs, actions, targets, ratios, noise, anchors, chunk=chunk)
                    self.assertTrue(all(d['sample_residual2_after'] <= d['sample_residual2_before']+1e-8 for d in info))
                torch.testing.assert_close(actual, expected, atol=4e-5, rtol=4e-5)

    def test_streamed_and_normal_real_parameter_update_alignment(self):
        for method in ('dual127p1', 'coeff128'):
            torch.manual_seed(41)
            model = TinyShared()
            clone = copy.deepcopy(model)
            obs, adv, ret = torch.randn(512, 4), torch.randn(512), torch.randn(512)
            actions = torch.arange(512)%3
            old = model(obs)[1].detach()+.1*torch.randn(512, 3)
            opt, opt2 = torch.optim.SGD(model.parameters(), lr=.5), torch.optim.SGD(clone.parameters(), lr=.5)
            rng = torch.get_rng_state()
            a = update(model, opt, obs, actions, adv, ret, old, method=method)
            end_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            b = update_large(clone, opt2, obs, actions, adv, ret, old, method=method, batch_size=512)
            self.assertTrue(torch.equal(end_rng, torch.get_rng_state()))
            for p, q in zip(model.parameters(), clone.parameters()):
                torch.testing.assert_close(p, q, rtol=5e-5, atol=5e-6)
            self.assertEqual(a['lr_after'], b['lr_after'])

    def test_large8192_real_loss_step_and_lr_inheritance(self):
        for method in ('dual127p1', 'coeff128'):
            torch.manual_seed(42)
            model = TinyShared()
            obs = torch.randn(8192, 4)
            old = model(obs)[1].detach()
            opt = torch.optim.SGD(model.parameters(), lr=.5)
            last = .5
            for index in range(2):
                m = update_large(model, opt, obs, torch.arange(8192)%3, torch.randn(8192),
                    torch.randn(8192), old, method=method, minibatch_index=index)
                self.assertEqual(m['lr_before'], last)
                last = m['lr_after']
                self.assertEqual(m['curvature_divisor'], 8192)
                self.assertEqual(m['loss_divisor'], 8192)
                self.assertEqual(m['anchors'], 127 if method == 'dual127p1' else 128)
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_fixed_profiles(self):
        self.assertEqual(len(PROFILES), 4)
        for name, (method, batch, endpoint) in PROFILES.items():
            self.assertIn(method, ('dual127p1', 'coeff128'))
            self.assertIn(batch, (512, 8192))
            self.assertEqual(endpoint % (8*batch), 0)
            self.assertEqual(4*(8*batch//batch), 32)
            self.assertEqual(endpoint//(8*batch), 1466 if batch == 512 else 229)

    def test_coefficient128_cli_defaults_to_ray_on(self):
        for profile in ('normal_128', 'large8192_128'):
            parser = build_parser(profile)
            self.assertEqual(parser.parse_args(['--out', 'unused']).residual_ray, 'on')
            self.assertEqual(parser.parse_args(['--out', 'unused', '--residual-ray', 'off']).residual_ray, 'off')
            self.assertEqual(parser.parse_args(['--out', 'unused', '--residual-ray', 'on']).residual_ray, 'on')
        for profile in ('normal_127p1', 'large8192_127p1'):
            self.assertEqual(build_parser(profile).parse_args(['--out', 'unused']).residual_ray, 'off')

    def test_no_ray_is_same_affine_restriction_without_scaling(self):
        _, factors, h = score_fixture(batch=160)
        targets = torch.stack((torch.randn(160), torch.ones(160)), 1)
        ratios = torch.stack((torch.linspace(.1, 3, 160), torch.ones(160)), 1)
        actual, info = solve_coeff128(factors, targets, ratios, torch.arange(128), .5, 160, ray_scale=False)
        for j in range(2):
            root = ratios[:, j].double().sqrt()
            m = root[:, None]*(h@h.T/160)*root[None, :] + .5*torch.eye(160).double()
            rhs = root*targets[:, j].double()
            x = rhs/.5
            x[:128] = torch.linalg.solve(m[:128, :128], rhs[:128]-m[:128, 128:]@x[128:])
            torch.testing.assert_close(actual[:, j].double(), x/root, atol=4e-5, rtol=4e-5)
            self.assertEqual(info[j]['ray_gamma'], 1.)
        model = TinyShared()
        obs = torch.randn(8192, 4)
        opt = torch.optim.SGD(model.parameters(), lr=.5)
        result = update_large(model, opt, obs, torch.arange(8192)%3, torch.randn(8192),
            torch.randn(8192), model(obs)[1].detach(), method='coeff128', ray_scale=False)
        self.assertFalse(result['coefficient_ray_scale'])
        self.assertTrue(all(d['ray_gamma'] == 1 for d in result['solve']))

    def test_streamed_convolution_coefficients(self):
        from test_algebra import TinyConv
        model = TinyConv()
        obs, actions, noise = torch.randn(160, 2, 4, 4), torch.arange(160)%3, torch.randn(160)
        targets = torch.stack((torch.randn(160), torch.ones(160)), 1)
        ratios = torch.stack((torch.linspace(.1, 3, 160), torch.ones(160)), 1)
        with LayerwiseIO(model) as io:
            values, logits = model(obs)
        factors = io.factors(logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)+2*noise*values)
        for ray in (False, True):
            a, _ = solve_coeff128(factors, targets, ratios, torch.arange(128), .5, 160, ray_scale=ray)
            b, _ = solve_streamed_coeff128(model, obs, actions, targets, ratios, noise, torch.arange(128), chunk=37, ray_scale=ray)
            torch.testing.assert_close(a, b, atol=5e-5, rtol=5e-5)
