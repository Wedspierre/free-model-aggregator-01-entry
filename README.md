# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Wedspierre/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Wedspierre/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One Node 22 server that serves the OpenAI Chat Completions API and the Anthropic Messages API — JSON and server-sent events — over ten free-tier providers on their own wires (OpenAI, Anthropic, Gemini). For every request it picks the best free model that meets the request's requirements, ranked by the sealed quality table, with a static plan that spreads the twenty requests across providers under their per-trial minimum budgets so no provider runs dry early. A provider that refuses (402, 429, 5xx) costs one call and the request is answered by the next best model that can serve it, honouring retry-after; a provider that fails three times running is set aside. Everything is deterministic — no wall clock, no unseeded randomness — so a replayed trial leaves the same export, and `node main.ts --list` prints the catalog byte-identical to the one registered with the env.

## Run it

```sh
node cli/makefun.mjs grade-local --kit . --bot ~/workspace/makefun/free-model-aggregator-repo --driver process
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ta01qs0h&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
