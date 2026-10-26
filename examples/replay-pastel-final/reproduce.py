from pathlib import Path
from mlstudio.research_export import replay_pack
if __name__ == "__main__":
    replay_pack(Path(__file__).resolve().parent)
