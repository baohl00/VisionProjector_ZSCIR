# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=redefined-outer-name,missing-module-docstring,g-importing-member,missing-function-docstring,g-bare-generic,g-doc-args,missing-class-docstring
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from typing import Any, List, Union
from tqdm import tqdm

import spacy
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from blip_caption import caption_generation, get_blip_model, split_caption 
import PIL
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def reference_image_dict(name = 'fiq'):
    if 'fiq' in name:
        file_name="./data/fiq/captions/cap.{ftype}.val.json"
        ftype=name[4:]
        data = json.load(open(file_name.format(ftype=ftype)))
        data = {i:data[i]["candidate"] for i in range(len(data))}
    elif name == 'cirr':
        data = json.load(open("./data/CIRR/cirr/captions/cap.rc2.test12.json"))
        data = {str(data_i["pairid"]): data_i["reference"] for data_i in data}
    else:
        data = json.load(open("./data/circo/annotations/test2.json"))
        data = {str(data_i["id"]):data_i["reference_img_id"] for data_i in data}
    return data        

@dataclass
class QueryExample:
    qid: str
    qtokens: np.ndarray
    qimage: np.ndarray
    target_iid: Union[int, str, List[int], List[str], None] # can be int or 
    retrieved_iids: List[Union[int, str]] # ranked by score, can be str (cirr) or int (circo)
    retrieved_scores: List[float] # ranked by order


@dataclass
class IndexExample:
    iid: Union[int, str]
    iimage: np.ndarray
    itokens: np.ndarray


@dataclass
class Dataset:
    name: str
    query_examples: List[QueryExample] = field(default_factory=list)
    k_range: List[int] = field(default_factory=lambda: [10, 50])
    # write_to_file_header: Dict[str, Any] = field(default_factory=dict)
    index_examples: List[IndexExample] = field(default_factory=list)

    def evaluate_recall(self):
        ret_dict = {k: [] for k in self.k_range}

        data = reference_image_dict(self.name)
        key = 0 
        #print(data)

        for q_example in self.query_examples:
            
            retrieved_iids = list(q_example.retrieved_iids)
            try:
                retrieved_iis.remove(data[key])
            except:
                pass
            retrieved_iids = retrieved_iids[:50] 
            key += 1

            assert len(retrieved_iids) > 0, "retrieved_iids is empty"
            for k in self.k_range:
                recalled = False
                if isinstance(q_example.target_iid, list):
                    for one_target_iid in q_example.target_iid:
                        if one_target_iid in retrieved_iids[:k]:
                            recalled = True
                elif isinstance(q_example.target_iid, int) or isinstance(q_example.target_iid, str):
                    if q_example.target_iid in retrieved_iids[:k]:
                        recalled = True
                else:
                    raise ValueError(f"target_iid is of type {type(q_example.target_iid)}")

                if recalled:
                    ret_dict[k].append(1)
                else:
                    ret_dict[k].append(0)
        # calculation
        total_ex = len(self.query_examples)
        ret_dict = {k: (sum(v) / total_ex) * 100 for k, v in ret_dict.items()}
        print("Recalls: ", ret_dict)

        return ret_dict


    def write_to_file(self, output_dir: str):
        if "cir" in self.name:
            if "cirr" in self.name:
                data = json.load(open("./data/CIRR/cirr/captions/cap.rc2.test12.json"))
                data = {str(data_i["pairid"]): data_i["reference"] for data_i in data}
            else:
                data = json.load(open("./data/circo/annotations/test2.json"))
                data = {str(data_i["id"]):data_i["reference_img_id"] for data_i in data}
        else:
            data = list()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        dict_to_write = dict()
        subset_dict_to_write = dict() 
        for q_example in tqdm(self.query_examples):
            #print(q_example.qid)
            qid = str(q_example.qid)
            #if self.name == "cirr":
            retrieved_iids = list(q_example.retrieved_iids)
            if data[qid] in retrieved_iids:
                retrieved_iids.remove(data[qid])
            dict_to_write[qid] = retrieved_iids[:50]
            subset_dict_to_write[qid] = retrieved_iids[:3]
            #dict_to_write[q_example.qid] = q_example.retrieved_iids[:50]
            #subset_dict_to_write[q_example.qid] = q_example.retrieved_iids[:3]
        output_file = os.path.join(output_dir, f"{self.name}_results.json")
        with open(output_file, "w") as f:
            if "circo" in output_file:
                json.dump(dict_to_write, f, indent=4)
            else:
                dict_to_write.update({
                    'version': 'rc2', 
                    'metric': 'recall'
                    }) 
                if "cirr" in output_file:
                    subset_dict_to_write.update({
                        'version': 'rc2', 
                        'metric': 'recall_subset'
                        })
                    subset_output_file = os.path.join(output_dir, f"subset_{self.name}_results.json")
                    with open(subset_output_file, "w") as f1:
                        json.dump(subset_dict_to_write, f1, indent=4)
                    print("Subset are written to file", subset_output_file)
                json.dump(dict_to_write, f)
        print("Results are written to file", output_file)


def process_img(image_path: str, size: int) -> np.ndarray:
    """Process a single image to 224x224 and normalize."""
    img = Image.open(image_path).convert("RGB")
    ima = jnp.array(img)[jnp.newaxis, ...] # [1, 224, 224, 3]
    ima = ima / (ima.max() + 1e-12)  # avoid 0 division
    ima = jax.image.resize(ima, (1, size, size, 3), method='bilinear')
    return np.array(ima)


def build_fiq_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    subtask = dataset_name.split("-")[1]
    queries = json.load(open(f"./data/fiq/captions/cap.{subtask}.val2.json"))
    index_img_ids = json.load(open(f"./data/fiq/image_splits/split.{subtask}.val.json"))
    index_image_folder = "./data/fiq/images"

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, index_img_id + ".png")
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)
    

    model, preprocess = get_blip_model(
            model_path = './model_base_capfilt_large.pth',
            blip_type = 'base',
            image_size = 384)
    nlp = spacy.load("en_core_web_sm")
    def preprocess_cap(caption):
        caption = caption.replace('t - shirt', 't-shirt')
        def replace_word(s, w):
            if w not in s:
                return s
            id = s.find(w)
            s = s[id+len(w):]
            return s 
        for word in ['man', 'men', 'women', 'woman', 'person']:
            w = f" {word} "
            caption = replace_word(caption, w)
        return caption

    def process_query_example(query):
        qid = query['candidate']
        qtext = " and ".join(query['captions'])
        qimage_path = os.path.join(index_image_folder, query['candidate'] + ".png")
        ima = process_img(qimage_path, 224)
        qcaption  = query["blip2_caption_opt"] 
        with open(qimage_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device) 
        with torch.no_grad():
            #qimg_caption = model.generate(image, sample=False, num_beams = 3, top_p = 0.9, max_length=20, min_length= 5) 
            #qimg_caption = qimg_caption[0].replace('t - shirt', 'tee')
            #img_exp = dict()
            #subject, attribute = split_caption(qcaption, nlp)  
            #print(img_exp)
            qcaption = preprocess_cap(qcaption) 
            qtext_new = qtext
            #qtext_new = qtext + ' DIFFERENT FROM ' + qcaption
            #qtext_new = "a phot of " + qtext + ' not like ' + qimg_caption
            qtokens = tokenizer(qtext_new)
            return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=query['target'], retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset


def build_circo_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    queries = json.load(open("./data/circo/annotations/test2.json"))
    coco_info = json.load(open("./data/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json"))
    index_img_ids = [img_info['id'] for img_info in coco_info['images']]
    index_image_folder = "./data/circo/COCO2017_unlabeled/unlabeled2017"

    def image_id2name(image_id):
        return str(image_id).zfill(12) + '.jpg'

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)
    
    model, preprocess = get_blip_model(
            model_path = './model_base_capfilt_large.pth',
            blip_type = 'base',
            image_size = 384)
    nlp = spacy.load("en_core_web_sm")

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['id']
        qimage_path = os.path.join(index_image_folder, image_id2name(query['reference_img_id']))
        with open(qimage_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device) 
        #with torch.no_grad():
        #    qimg_caption = model.generate(image, sample=False, num_beams = 3, top_p = 0.9, max_length=20, min_length= 5) 
        #    qimg_caption = qimg_caption[0]
        #    img_exp = dict()
        #    subject, attribute = split_caption(qimg_caption, nlp) 
        #    if subject != attribute:
        #        img_exp["img_id"] = qid 
        #        img_exp["subject"] = subject
        #        img_exp["attribute"] = attribute
        #    else: 
        #        img_exp["subject"] = subject
            #print(img_exp)
            #qtext_new = 'A photo ' + qtext + ' and different from ' + qimg_caption
        #qtext = f"{query['gemma_generated_query']} AND {query['relative_caption']}" # but {query['gemma_generated_query']}"
        #qtext = f"{query['relative_caption']}" #  SAME {query['shared_concept']}"
        qtext = f"{query['shared_concept']} but {query['relative_caption']}"
        #qtext = f"{query['relative_caption']} but {query['shared_concept']}"
        #qtext = f"TOTALLY {query['relative_caption']} SAME {qimg_caption}"
        ima = process_img(qimage_path, 224)
        qtokens = np.array(tokenizer(qtext))
        # circo test does not provide target id.
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset

def build_cirr_dataset(dataset_name: str, tokenizer: Any) -> Dataset:
    eval_dataset = Dataset(dataset_name)
    queries = json.load(open("./data/CIRR/cirr/captions/cap.rc2.test12.json"))
    coco_info = json.load(open("./data/CIRR/cirr/image_splits/split.rc2.test1.json"))
    index_img_ids = coco_info.keys() # [img_info['id'] for img_info in coco_info['images']]
    index_image_folder = "./data/CIRR/test1"

    def image_id2name(image_id):
        return image_id + '.png'

    null_tokens = tokenizer("")  # used for index example
    null_tokens = np.array(null_tokens)
    
    model, preprocess = get_blip_model(
            model_path = './model_base_capfilt_large.pth',
            blip_type = 'base',
            image_size = 384)
    nlp = spacy.load("en_core_web_sm")

    def process_index_example(index_img_id):
        img_path = os.path.join(index_image_folder, image_id2name(index_img_id))
        ima = process_img(img_path, 224)
        return IndexExample(iid=index_img_id, iimage=ima, itokens=null_tokens)

    def process_query_example(query):
        qid = query['pairid']
        qimage_path = os.path.join(index_image_folder, image_id2name(query['reference']))
        with open(qimage_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device) 
        #with torch.no_grad():
        #    qimg_caption = model.generate(image, sample=False, num_beams = 3, top_p = 0.9, max_length=15, min_length= 5) 
        #    qimg_caption = qimg_caption[0]
        #    img_exp = dict()
        #    subject, attribute = split_caption(qimg_caption, nlp) 
        #    if subject != attribute:
        #        img_exp["img_id"] = qid 
        #        img_exp["subject"] = subject
        #        img_exp["attribute"] = attribute
        #    else: 
        #        img_exp["subject"] = subject
            #print(img_exp)
            #qtext_new = 'A photo ' + qtext + ' and different from ' + qimg_caption
        qtext = query['caption']
        #qtext = f"{query['caption']} WITH {query['gemma_generated_query']}" # DIFFERENT FROM {qimg_caption}"
        ima = process_img(qimage_path, 224)
        qtokens = np.array(tokenizer(qtext))
        # circo test does not provide target id.
        return QueryExample(qid=qid, qtokens=qtokens, qimage=ima, target_iid=0, retrieved_iids=[], retrieved_scores=[])

    with ThreadPoolExecutor() as executor:
        print("Preparing index examples...")
        index_example_futures = {executor.submit(process_index_example, index_img_id): index_img_id for index_img_id in index_img_ids}

        with tqdm(total=len(index_img_ids), desc="Index examples") as progress:
            for future in as_completed(index_example_futures):
                index_example = future.result()
                eval_dataset.index_examples.append(index_example)
                progress.update(1)

        print("Prepared index examples.")

        print("Preparing query examples...")
        query_futures = {executor.submit(process_query_example, query): query for query in queries}

        with tqdm(total=len(queries), desc="Query examples") as progress:
            for future in as_completed(query_futures):
                q_example = future.result()
                eval_dataset.query_examples.append(q_example)
                progress.update(1)

    return eval_dataset


