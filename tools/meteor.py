import sys
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

if __name__ == "__main__":
    pred_path = sys.argv[1]
    data_path = sys.argv[2]
    with open(pred_path, "r") as file:
        pred = file.readlines()

    with open(data_path, "r") as file:
        target = file.readlines()

    zip_list = zip(target, pred)

    
    #for t,p in zip_list:
     # print([nltk.tokenize.word_tokenize(p.lower())])

    scores = [nltk.meteor([nltk.tokenize.word_tokenize(t.lower())], nltk.tokenize.word_tokenize(p.lower())) for t,p in zip_list]
    print(sum(scores)/len(scores))
