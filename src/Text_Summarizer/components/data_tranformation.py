{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3721f555",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "from Text_Summarizer.logging import logger\n",
    "from transformers import AutoTokenizer\n",
    "from datasets import load_dataset, load_from_disk\n",
    "from Text_Summarizer.entity import DataTransformationConfig"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f5279135",
   "metadata": {},
   "outputs": [],
   "source": [
    "class DataTransformation:\n",
    "    def __init__(self, config: DataTransformationConfig):\n",
    "        self.config = config\n",
    "        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)\n",
    "\n",
    "\n",
    "    \n",
    "    def convert_examples_to_features(self,example_batch):\n",
    "        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )\n",
    "        \n",
    "        with self.tokenizer.as_target_tokenizer():\n",
    "            target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )\n",
    "            \n",
    "        return {\n",
    "            'input_ids' : input_encodings['input_ids'],\n",
    "            'attention_mask': input_encodings['attention_mask'],\n",
    "            'labels': target_encodings['input_ids']\n",
    "        }\n",
    "    \n",
    "\n",
    "    def convert(self):\n",
    "        dataset_samsum = load_from_disk(self.config.data_path)\n",
    "        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)\n",
    "        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,\"samsum_dataset\"))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
