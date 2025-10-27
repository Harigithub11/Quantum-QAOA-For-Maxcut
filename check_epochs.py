import torch
import os

checkpoint_dir = "models/checkpoints"
checkpoints = [
    "checkpoint_epoch_000.pth",
    "checkpoint_epoch_001.pth", 
    "checkpoint_epoch_002.pth"
]

for ckpt_name in checkpoints:
    ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print("\n" + "="*60)
        print(f"📊 {ckpt_name.replace('.pth', '').upper()}")
        print("="*60)
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Get metrics dictionary
        metrics = checkpoint.get('metrics', {})
        model_info = checkpoint.get('model_info', {})
        timestamp = checkpoint.get('timestamp', 'N/A')
        
        # Display all available metrics
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Timestamp: {timestamp}")
        
        print(f"\n📈 Training Metrics:")
        train_loss = metrics.get('train_loss', 'N/A')
        train_acc = metrics.get('train_acc', 'N/A')  # Changed from train_accuracy
        print(f"  - Train Loss: {train_loss:.6f}" if isinstance(train_loss, (int, float)) else f"  - Train Loss: {train_loss}")
        print(f"  - Train Accuracy: {train_acc:.4f}%" if isinstance(train_acc, (int, float)) else f"  - Train Accuracy: {train_acc}")
        
        print(f"\n✅ Validation Metrics:")
        val_loss = metrics.get('val_loss', 'N/A')
        val_acc = metrics.get('val_acc', 'N/A')  # Changed from val_accuracy
        best_val_acc = metrics.get('best_val_acc', metrics.get('best_val_accuracy', 'N/A'))
        print(f"  - Val Loss: {val_loss:.6f}" if isinstance(val_loss, (int, float)) else f"  - Val Loss: {val_loss}")
        print(f"  - Val Accuracy: {val_acc:.4f}%" if isinstance(val_acc, (int, float)) else f"  - Val Accuracy: {val_acc}")
        print(f"  - Best Val Accuracy: {best_val_acc:.4f}%" if isinstance(best_val_acc, (int, float)) else f"  - Best Val Accuracy: {best_val_acc}")
        
        print(f"\n⚙️ Training Config:")
        lr = metrics.get('learning_rate', 'N/A')
        print(f"  - Learning Rate: {lr}")
        
        print(f"\n📦 Checkpoint Contents:")
        print(f"  Main Keys: {', '.join(checkpoint.keys())}")
        print(f"  Metrics Keys: {', '.join(metrics.keys()) if metrics else 'None'}")
        print(f"  Model Info: {model_info}")
    else:
        print(f"\n❌ {ckpt_name} not found")

print("\n" + "="*60)
