"""Step 0: download the nine dataset CSVs into data/ and check their SHA-256.

    python scripts/download_data.py

The files are taken from the PepMNet repository at a fixed commit; they are the
exact files the published results were trained on. See data/README.md for their
original sources, which should be cited.
"""

import hashlib
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
URL = "https://raw.githubusercontent.com/danielgarzonotero/PepMNet/2986c9fac2c188df02f306607297067a91ebdbb1/data/RT/{}.csv"
SHA256 = {
    "hela_mod3": "602320c45d53d6d8d9c2f6bb941d5030d016634841806f18c90fa44194fd78b6",
    "hela_mod": "4cb4d2d9df19b2a4d4d28e350aedcf90dfe6cabe143bf49fc05b8a0ae4eaa351",
    "yeast_unmod": "a92d9a134e59b3f62b6ea3b4901da7f0d331d332ca51ca1c1d4a1eb84c35ed25",
    "scx": "2f094cd0093b81ebaacbdeec2c531bcb9aa808c9d9f349ab9845491abdc3e939",
    "luna_hilic": "bfedfd20d5511cd40c178fbf202e21b8a661c5430ad71e3810cf55eb343eeb6a",
    "luna_silica": "808f7c7c67a4161e9f6da42a0f2a32ac8bd1dbf28159d65652d6d0360818a49e",
    "atlantis_silica": "74bc27752efe4fa430b9df9bcac16ebd2d48478d6a5f8dfdc64e62e19ee64eac",
    "xbridge_amide": "3643961d0b4612e5923fa0231881274e3c283581758702f17a49c95ec72b20dc",
    "misc_dia": "cfed677e8baabb716888476a6ab1c2a7ec269758cdbf4909bf7c510257e0c786",
}


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, expected in SHA256.items():
        path = DATA / f"{name}.csv"
        if not path.exists():
            with urllib.request.urlopen(URL.format(name)) as response:
                path.write_bytes(response.read())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"{path}: sha256 {actual} does not match {expected}")
        print(f"ok  {path.name}")


if __name__ == "__main__":
    main()
