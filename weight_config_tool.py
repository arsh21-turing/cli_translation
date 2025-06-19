import argparse
import json
import os
from config_manager import ConfigManager

def setup_argparser():
    parser = argparse.ArgumentParser(description='Quality Weights Configuration Tool')
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List weights command
    list_parser = subparsers.add_parser('list', help='List all current weights')
    list_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    # Set weight command
    set_parser = subparsers.add_parser('set', help='Set a specific weight')
    set_parser.add_argument('name', help='Name of the weight')
    set_parser.add_argument('value', type=float, help='Value to set (0.0-1.0)')
    
    # Update from file command
    update_parser = subparsers.add_parser('update-from-file', help='Update weights from JSON file')
    update_parser.add_argument('file', help='Path to JSON file with weights')
    
    # Save weights command
    save_parser = subparsers.add_parser('save', help='Save current weights to a config file')
    save_parser.add_argument('--path', default='~/.tqa/config.json', help='Path to save config file')
    
    # Reset weights command
    reset_parser = subparsers.add_parser('reset', help='Reset weights to default values')
    
    return parser

def main():
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    if args.command == 'list':
        # List all weights
        weights = config_manager.get_quality_weights()
        
        if args.json:
            print(json.dumps(weights, indent=2))
        else:
            print("Current quality score weights:")
            print("-" * 30)
            
            # Group weights by category
            categories = {
                "Embedding Metrics": ["embedding_similarity", "length_ratio_penalty", "embedding_metrics_weight"],
                "Alignment Metrics": ["alignment_score", "recurring_pattern_penalty", "position_pattern_penalty", "alignment_metrics_weight"],
                "Groq Simple Metrics": ["groq_score", "groq_simple_metrics_weight"],
                "Groq Detailed Metrics": ["accuracy", "fluency", "terminology", "style", "groq_detailed_metrics_weight"]
            }
            
            for category, weight_names in categories.items():
                print(f"\n{category}:")
                for name in weight_names:
                    if name in weights:
                        print(f"  {name}: {weights[name]:.2f}")
            
            # Print any uncategorized weights
            uncategorized = [name for name in weights if not any(name in names for names in categories.values())]
            if uncategorized:
                print("\nOther Weights:")
                for name in uncategorized:
                    print(f"  {name}: {weights[name]:.2f}")
    
    elif args.command == 'set':
        # Set a specific weight
        name = args.name
        value = args.value
        
        # Validate value
        if value < 0 or value > 1:
            print(f"Error: Weight value must be between 0.0 and 1.0")
            return 1
        
        config_manager.set_weight(name, value)
        print(f"Set {name} = {value:.2f}")
        
    elif args.command == 'update-from-file':
        # Update weights from a JSON file
        file_path = args.file
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found")
            return 1
            
        try:
            with open(file_path, 'r') as f:
                weights = json.load(f)
                
            # Validate weights
            invalid_weights = [name for name, val in weights.items() 
                             if not isinstance(val, (int, float)) or val < 0 or val > 1]
            
            if invalid_weights:
                print(f"Error: Invalid weight values for: {', '.join(invalid_weights)}")
                return 1
                
            config_manager.update_weights(weights)
            print(f"Updated {len(weights)} weights from {file_path}")
            
        except json.JSONDecodeError:
            print(f"Error: File {file_path} is not valid JSON")
            return 1
        except Exception as e:
            print(f"Error updating weights: {str(e)}")
            return 1
    
    elif args.command == 'save':
        # Save weights to a config file
        path = os.path.expanduser(args.path)
        
        success = config_manager.save(path)
        if success:
            print(f"Configuration saved to {path}")
        else:
            print(f"Error saving configuration to {path}")
            return 1
    
    elif args.command == 'reset':
        # Reset weights to default
        config_manager.reset_weights_to_default()
        print("Weights reset to default values")
    
    else:
        # No command specified
        parser.print_help()
    
    return 0

if __name__ == '__main__':
    exit(main()) 