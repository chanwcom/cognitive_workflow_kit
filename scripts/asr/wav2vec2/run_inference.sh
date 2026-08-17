#python wav2vec2_inference.py --checkpoint_dir=/mnt/data/home/chanwcom/models/alpha_0p02_beta_0p1_experiment/checkpoint-2000 --decoder tf_beam_search --beam_size 20 --vocab_size 32  
#--debug_tf_decoder  
#python wav2vec2_inference.py --checkpoint_dir=/mnt/data/home/chanwcom/models/shc_2000steps_alpha_0p0_beta_0p0_unigram_32/checkpoint-2000 --decoder tf_beam_search --beam_size 40 --vocab_size 32  #--debug_tf_decoder  
#python wav2vec2_inference.py --checkpoint_dir=/mnt/data/home/chanwcom/models/shc_2000steps_alpha_0p0_beta_0p0_unigram_32/checkpoint-2000 --decoder beam_search --beam_size 40 --vocab_size 32
python wav2vec2_inference.py --checkpoint_dir=/mnt/data/home/chanwcom/models/alpha_0p02_beta_0p1_experiment/checkpoint-2000 --decoder beam_search --beam_size 20 --vocab_size 32  
