import sys


def main() -> None:
    system_major = sys.version_info.major
    system_minor = sys.version_info.minor

    if system_major != 3 or system_minor < 8:
        raise TypeError(f"This project requires Python >= 3.8. Found: Python {system_major}.{system_minor}")
    print("Development environment passes all tests!")


if __name__ == "__main__":
    main()
