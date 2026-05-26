# Python file that mounts and loads data
import os
import pandas as pd
from pathlib import Path

def mount_drive(base_path: str = None) -> Path:
    try:
        # Check if rclone Google Drive is mounted
        gdrive_mount = Path('/data/krosen5/gdrive-all')

        if gdrive_mount.exists() and gdrive_mount.is_dir():
            print(f"✓ Found Google Drive mount at {gdrive_mount}")

            # Test if mount is actually working
            try:
                list(gdrive_mount.iterdir())
                print("✓ Google Drive mount is accessible")
            except OSError:
                print("✗ Google Drive mount exists but is not accessible")
                print("Try: systemctl --user restart rclone-gdrive.service")
                raise

            # Navigate to the data directory
            if base_path:
                root = gdrive_mount / base_path
            else:
                root = gdrive_mount

            if root.exists():
                print(f"✓ Data directory found: {root}")
                os.chdir(root)
                return root
            else:
                print(f"✗ Data directory not found: {root}")
                print(f"Available directories in {gdrive_mount}:")
                for item in gdrive_mount.iterdir():
                    if item.is_dir():
                        print(f"  - {item.name}")
                raise FileNotFoundError(f"Data directory {root} not found")

        else:
            print("✗ Google Drive not mounted at /data/krosen5/gdrive-all")
            print("Please run the setup script first or mount manually:")
            print("  systemctl --user start rclone-gdrive.service")
            raise FileNotFoundError("Google Drive not mounted")

    except Exception as e:
        print(f"Error accessing Google Drive: {e}")
        # Fallback to current directory
        print("Falling back to current directory")
        return Path(os.getcwd())

def load_data(root: Path) -> dict:
    data = {
        'mag': pd.read_csv(root / 'proceedings.csv', dtype={'proceeding_id':str,'mag_num':str}).rename(columns={'section': 'magistrate_section'}),
        'cdc': pd.read_csv(root / 'cdc_proceedings.csv', dtype={'proceeding_id':str,'case_id':str}).rename(columns={'section': 'cdc_section'}),
        'defendants': pd.read_csv(root / 'defendants_to_use_for_project.csv', dtype={'defendant_id':str}),
        'mag_charges': pd.read_csv(root / 'docket_charges.csv', dtype={'mag_num':str}),
        'cdc_charges': pd.read_csv(root / 'cdc_docket_charges.csv', dtype={'case_id':str}),
        'mag_dockets': pd.read_csv(root / 'dockets.csv', dtype={'mag_num':str,'defendant_id':str}),
        'cdc_dockets': pd.read_csv(root / 'cdc_dockets.csv', dtype={'case_id':str,'defendant_id':str}),
        'charges': pd.read_csv(root / 'charges.csv', dtype=str),
        'charge_details': pd.read_csv(root / 'charge_details.csv', dtype=str)  # Add this
    }

    data['mag_dockets']['case_id'] = data['mag_dockets']['case_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) else x)
    return data