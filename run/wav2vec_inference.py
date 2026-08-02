# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os

# Third-party imports
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoProcessor
from torch.utils import data
import tensorflow as tf
import torch
import transformers
from torchaudio.models import decoder
import evaluate
import time

# Custom imports
from data.format import speech_data_helper
from data.operation import text_codec
from data.operation import text_codec_params

# Prevents Tensorflow from using the entire GPU memory.
#
# Since we use Tensorflow and Pytorch simultaneously, Tensorflow shouldl not
# occupy the entire memory. Instead of allocating the entire GPU memory, GPU
# memory allocated to Tensorflow grows based on its need. Refer to the
# following website for more information:
# https://www.tensorflow.org/guide/gpu
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized.
        print(e)

db_top_dir = "/mnt/data/database/libri_speech/tfrecord"
test_top_dir = db_top_dir


# yapf: disable
op = speech_data_helper.SpeechDataToWave()
test_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(test_top_dir, "test-clean.tfrecord-*")),
              compression_type="GZIP")
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(op.process)
# yapf: enable

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base", ignore_mismatched_sizes=True)
resource_top="/mnt/data/home/chanwcom/local_repository/cognitive_workflow_kit/run"

#params = text_codec_params.TextCodecParams(
#        os.path.join(resource_top, "model_unigram_128.model"),
#        text_codec_params.ProcessingMode.DECODING, False, False)
#codec = text_codec.SentencePieceTextCodec(params)

tokenizer = transformers.AutoTokenizer.from_pretrained("facebook/wav2vec2-base")

class IterDataset(data.IterableDataset):

    def __init__(self, tf_dataset):
        self._dataset = tf_dataset

    def __iter__(self):
        for data in self._dataset:
            output = {}
            output["input_values"] = torch.tensor(tf.squeeze(data[0]["SEQ_DATA"]).numpy(), device="cuda")
            output["input_length"] = torch.tensor(tf.squeeze(data[0]["SEQ_LEN"]).numpy(), device="cuda")
            output["labels"] = [data.numpy().decode("unicode-escape")  for  data in data[1]["SEQ_DATA"]]

            yield (output)


processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
pytorch_test_dataset = IterDataset(test_dataset)


#model = transformers.Wav2Vec2ForCTC.from_pretrained(
#    "/mnt/data/home/chanwcom/experiment/asr_libri_light_1hr_els_new_0p03_large_1m4/checkpoint-2000").to("cuda").to(torch.bfloat16)
model = transformers.Wav2Vec2ForCTC.from_pretrained("/mnt/data/home/chanwcom/experiment/asr_wav2vec2_base_libri_light_1hr_elsa_two_0p0_1000_2000_00/checkpoint-8000").to("cuda").to(torch.bfloat16)
#, ignore_mismatched_sizes=True, vocab_size=133) #, num_labels=133, ignore_mismatched_sizes=True)
#model.config.vocab_size = 133
#model = transformers.Wav2Vec2ForCTC.from_pretrained("/mnt/data/home/chanwcom/experiment/asr_libri_light_1hr_model_08/checkpoint-1500", vocab_size=28, ignore_mismatched_sizes=True)
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,# "/mnt/data/home/chanwcom/experiment/asr_libri_light_1hr_model_11/checkpoint-2000",
    feature_extractor=processor.feature_extractor,
    tokenizer=tokenizer,
    device_map="auto"
)


vocab = processor.tokenizer.vocab
tokens = sorted(vocab, key=lambda inputs : vocab[inputs])
beam_search_decoder = decoder.ctc_decoder(
    lexicon=None,
    tokens=tokens,
    lm=None,
    nbest=1,
    beam_size=50,
    blank_token="<pad>"
)



ref_list = []
hyp_list = []

start_time = time.time()

for index, data in enumerate(pytorch_test_dataset):
    ref = data["labels"]
    print(f"REF: {ref}")

    #if index == 100:
    #    break

    ref_list.extend(ref)

    if 1:
        outputs = transcriber([data["input_values"].to("cpu").numpy()])
        print(f"HYP: {outputs}")
        hyp_list.extend([output["text"]  for output in outputs])


    if 0:
        feature_output = processor.feature_extractor(
            data["input_values"], sampling_rate=16000)
        model_output = model.forward(
            torch.tensor([feature_output["input_values"][0]], device="cuda").to(torch.bfloat16))
        outputs = beam_search_decoder(model_output.logits.to(torch.float32).to("cpu"))

        hyp = ["".join([tokens[element] for element in output[0].tokens]).
               replace("|", " ").strip() for output in outputs]
        print (hyp)
        hyp_list.extend(hyp)

    if 0:
        logits = torch.transpose(model_output.logits, 1, 0)

        # Replaces the location of <PAD> which is also used for blank
        logits_new = logits.detach().clone()
        logits_new[:, :, 0] = logits[:, :, -1]
        logits_new[:, :, -1] = logits[:, :, 0]

        tf_tensor = tf.convert_to_tensor(logits_new.detach().numpy())

        outputs = tf.nn.ctc_beam_search_decoder(
            tf_tensor, [tf_tensor.shape[0]], 32, 1)
        outputs = outputs[0][0].values

        outputs = "".join([tokens[element] for element in outputs]).replace("|", " ").strip()
        print (f"HYP: {outputs}")
        hyp_list = [
            "".join([tokens[element] for element in single_output[0].tokens])
                .replace("|", " ").strip() for single_output in hyp_list]

        hyp_list.append(outputs)

#        inputs = {}
#        inputs["SEQ_DATA"] = [outputs.numpy().tolist()]
#        inputs["SEQ_LEN"] = outputs.shape[0]
#
#        outputs = codec.process(inputs)
#        hyp_string = outputs["SEQ_DATA"][0]
#        hyp_list.append(hyp_string)
#        print (f"HYP: {hyp_string}")



print (f"Elapsed time during the experiment: {time.time() - start_time:.2f} s")
wer = evaluate.load("wer")


result = wer.compute(references=ref_list, predictions=hyp_list)
print (result)
