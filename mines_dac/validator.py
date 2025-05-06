def validate_spectra_file(file_path: str) -> bool:
    try:
        with open(file_path, 'r') as f:
            lines = [next(f).strip() for _ in range(3)]
            if lines[1].lower() != "counts" or "SciMode" not in lines[2]:
                return False

            parts = next(f).strip().split()
            if len(parts) != 2:
                return False
            float(parts[0])
            float(parts[1])
            return True
    except Exception as e:
        print(f"Validation error: {e}")
        return False
