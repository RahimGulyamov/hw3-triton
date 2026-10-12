from pathlib import Path

root = Path("/results/model_analyzer/model_repository")
for config in sorted(root.glob("*/config.pbtxt")):
    print(config.parent.name)
