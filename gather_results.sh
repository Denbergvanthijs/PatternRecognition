# GAN results
python evaluate.py --path-predicted ./unpadded/gan/output_pretrained --path-ground-truth ./input/218/test/original --output-path ./results/gan/pretrained.json
python evaluate.py --path-predicted ./unpadded/gan/output_100 --path-ground-truth ./input/218/test/original --output-path ./results/gan/100.json
python evaluate.py --path-predicted ./unpadded/gan/output_1000 --path-ground-truth ./input/218/test/original --output-path ./results/gan/1000.json
python evaluate.py --path-predicted ./unpadded/gan/output_2500 --path-ground-truth ./input/218/test/original --output-path ./results/gan/2500.json

# CNN results
# python evaluate.py --path-predicted ./unpadded/cnn/output_pretrained --path-ground-truth ./input/218/test/original --output-path ./results/cnn/pretrained.json
# python evaluate.py --path-predicted ./unpadded/cnn/output_100 --path-ground-truth ./input/218/test/original --output-path ./results/cnn/100.json
# python evaluate.py --path-predicted ./unpadded/cnn/output_1000 --path-ground-truth ./input/218/test/original --output-path ./results/cnn/1000.json
# python evaluate.py --path-predicted ./unpadded/cnn/output_2500 --path-ground-truth ./input/218/test/original --output-path ./results/cnn/2500.json

# Transformer results
# python evaluate.py --path-predicted ./unpadded/transformer/output_pretrained --path-ground-truth ./input/218/test/original --output-path ./results/transformer/pretrained.json
# python evaluate.py --path-predicted ./unpadded/transformer/output_100 --path-ground-truth ./input/218/test/original --output-path ./results/transformer/100.json
# python evaluate.py --path-predicted ./unpadded/transformer/output_1000 --path-ground-truth ./input/218/test/original --output-path ./results/transformer/1000.json
# python evaluate.py --path-predicted ./unpadded/transformer/output_2500 --path-ground-truth ./input/218/test/original --output-path ./results/transformer/2500.json
