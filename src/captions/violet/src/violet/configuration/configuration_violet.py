import torch

class VioletConfig:

  """
  Configuration for Violet Pipeline.
  Contains default parameters that can be used globally in the pipeline.
  """

  # General settings
  CHECKPOINT_DIR = "/content/drive/MyDrive/Violet_checkpoint_0.pth"  # Path to the pretrained model
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device for computation (GPU/CPU)
  TOKENIZER_NAME = "UBC-NLP/Jasmine-350M"  # Tokenizer model name
  PROCESSOR_NAME = "openai/clip-vit-large-patch14"  # Processor model name
  
  # Model settings
  ENCODER_LAYERS = 3  # Number of layers in the visual encoder
  DECODER_LAYERS = 12  # Number of layers in the decoder
  TAU = 0.3  # Temperature parameter for the softmax function

  # Generation settings
  MAX_LENGTH = 40 # Maximum length of generated sequences
  BEAM_SIZE = 5 # Beam size for beam search decoding
  OUT_SIZE = 3 # Number of output sequences to generate