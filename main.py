
import sys

class Main:
    def __init__(self, ratings_file, content_file, targets_file):
        self.ratings_file = ratings_file
        self.content_file = content_file
        self.targets_file = targets_file

   

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 main.py ratings.jsonl content.jsonl targets.csv > submission.csv")
        sys.exit(1)

    ratings_file = sys.argv[1]
    content_file = sys.argv[2]
    targets_file = sys.argv[3]

    main = Main(ratings_file, content_file, targets_file)
  
