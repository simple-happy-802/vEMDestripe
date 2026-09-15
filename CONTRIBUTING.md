# Contributing

Please open an issue before a large change. Bug reports should include the command, environment, input shape/bit depth, reference protocol, complete traceback, and a minimal reproducible example that can be shared legally.

For code changes:

1. preserve strict checkpoint compatibility unless the change is explicitly versioned as a new architecture;
2. add or update tests;
3. run `python -m compileall -q .` and `python -m pytest -q`;
4. document changes to reference construction, tiling, normalization, or evaluation because each can alter reported image quality;
5. do not commit private data, large volumes, checkpoints, or generated results.

Scientific contributions should state whether they require retraining and should report uncertainty across independent seeds when making comparative claims.

