import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth",
            type = str,
            default = "./data/circo/annotations/test.json")
    parser.add_argument("--predict",
            type = str, 
            default = "./output/circo_large/circo_results.json")
    args = parser.parse_args()

    # Read json file circo/annotations/test.json
    with open(args.ground_truth, 'r') as file:
        data = json.load(file)

    # New data from beforehand data with the keys as the id and the values as the reference_img_id
    new_data = {}
    for i in data:
        new_data[i['id']] = i['reference_img_id']

    # Read json file circo_base/circo_results.json 
    with open(args.predict, 'r') as file:
        results = json.load(file) 
    
    # Change the keys in results to int     
    results = {int(k): v for k, v in results.items()}

    # Check if the reference_img_id is the same as the corresponding list in the results
    correct = 0
    for i in new_data.keys():
        if new_data[i] in results[i]:
            correct += 1

    print(correct/len(new_data))
