const fs = require('fs');
const path = '/Users/user/.codex/visualizations/2026/07/20/019f8055-8ac8-7b41-a529-37dbaa4704aa/procgen-rat-vs-ppo-curves.html';
const html = fs.readFileSync(path, 'utf8');
const matches = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if (matches.length !== 1) throw new Error(`expected one script, got ${matches.length}`);
new Function(matches[0][1]);
if (!html.includes('id="pc-grid"')) throw new Error('missing grid target');
if (html.includes('__CURVE_DATA__')) throw new Error('unfilled data placeholder');
console.log('visual syntax and targets: PASS');
