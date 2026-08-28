# Task64 BigFish partial terminal evidence

BigFish job19531929 naturally completed `COMPLETED/0:0` after00:45:43 on
node823. Root is PASS/rc0 at exact2,007,040 with reward1.97 versus immutable
Paper9.28 and Task63 BigFish5.08. It contains49 progress rows and15,680 complete
records. The remote checkpoint is a regular non-symlink file of3,766,013 bytes,
mode644; only stat metadata is retained.

Model-free hashes: progress
`1c3b8c87b2a17e6fe75d07264932111033447098a139868dc2671134fc57944f`,
trace `56b4cc39f0b72a04757bf512a28686a8c943a8a1cd515d2df52a49b6abacb1b8`,
stdout `d8b9fa520b61bbb5fd6006ba75d3c8a938bb72dade811a7b62b80dba1cda0f3e`,
stderr `fc172de29c0f61a3c19b437be0f5ee1b24a28c9c2fe0a83b9bf516b371b63e47`.
Focused hard-error matches are zero.

The frozen Task64 aggregate passes all records. Overall medians:

- actor metric norm/energy `.2418002337/.0923167430`;
- post-inverse full actor norm/projection `.7359012961/.9220500886`;
- shared actor norm/projection `.7339004576/.9193134308`;
- full cosine `-.1213011183`, cancellation `.7502087951`;
- clip rate `.7219387755`.

Early/middle/late full actor norm medians are
`.7452625632/.7485859990/.7152255177`; projection medians are
`.9125550389/.9332025051/.9186274409`. Thus curvature4 makes the raw metric
critic-heavy but the post-inverse installed-direction decomposition strongly
actor-heavy. Compared with Task63 BigFish, it reduces reward5.08 to1.97 and
does not increase the critic direction contribution.

BossFight19531930, CaveFlyer19531931 and naturally-started CoinRun19531932
remain RUNNING and untouched. Task64 remains `SCIENCE_RUNNING_PARTIAL_TERMINAL`.
