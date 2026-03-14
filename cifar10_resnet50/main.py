import train
import test

if __name__ == "__main__":
    print("Starting training...")
    train.main()
    print("\nEvaluating on test set...")
    test.main()