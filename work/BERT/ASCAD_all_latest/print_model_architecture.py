import torch
from train_ascad_bert import BERT_SCA_Model, BertModel

def print_model_architecture(model_path):
    # Initialize the model
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BERT_SCA_Model(bert_model, trace_length=1400)  # Using 1400 as it's common for ASCAD
    
    # Load the state dict
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Print checkpoint keys to see what information is stored
    print("\nCheckpoint Information:")
    print("=" * 50)
    if isinstance(checkpoint, dict):
        print("Checkpoint contains the following keys:")
        for key in checkpoint.keys():
            print(f"- {key}")
        
        # Try to find seed information
        if 'seed' in checkpoint:
            print(f"\nRandom seed used: {checkpoint['seed']}")
        elif 'config' in checkpoint and 'seed' in checkpoint['config']:
            print(f"\nRandom seed used: {checkpoint['config']['seed']}")
        else:
            print("\nNo random seed information found in checkpoint")
            
        # Load model state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Checkpoint is a direct state dict, no additional information available")
        model.load_state_dict(checkpoint)
    
    # Print model architecture
    print("\nModel Architecture:")
    print("=" * 50)
    print(model)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Parameters:")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Print layer details
    print("\nLayer Details:")
    print("=" * 50)
    for name, module in model.named_children():
        print(f"\n{name}:")
        print(f"Type: {type(module).__name__}")
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            print(f"Input features: {module.in_features}")
            print(f"Output features: {module.out_features}")

if __name__ == "__main__":
    model_path = "/root/alper/jupyter_transfer/downloaded_data/work/BERT/ASCAD_all_latest/models/bestmodel/bert_sca_ascad-variable-desync50_89Trace.pth"
    print_model_architecture(model_path) 