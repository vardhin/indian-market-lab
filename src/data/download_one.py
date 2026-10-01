from pathlib import Path
import subprocess

DATE = "20260925"

url = (
    "https://nsearchives.nseindia.com/content/cm/"
    f"BhavCopy_NSE_CM_0_0_0_{DATE}_F_0000.csv.zip"
)

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

out_name = f"bhavcopy_{DATE}.zip"

cmd = [
    "aria2c",
    "-x", "8",
    "-s", "8",
    "-k", "1M",
    "--allow-overwrite=true",
    "--auto-file-renaming=false",
    "--dir", str(out_dir),
    "--out", out_name,
    url,
]

print("Downloading:")
print(url)

subprocess.run(cmd, check=True)

print("Saved:", out_dir / out_name)
