from src.preprocessing.data_loader import load_dataset

def main():
    df = load_dataset()
    print(df.columns.tolist())
    print(df[[ "target_t2", "target_t23", "target_t234" ]].head())

if __name__ == "__main__":
    main()