import json
import sys
import os

def run_notebook(nb_path):
    """
    Reads a Jupyter notebook file, extracts all python code cells, 
    and executes them in sequence in the current global namespace.
    """
    print(f"\n==================================================")
    print(f"RUNNING NOTEBOOK: {nb_path}")
    print(f"==================================================")
    
    if not os.path.exists(nb_path):
        print(f"Error: Notebook not found at {nb_path}")
        return False
        
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    # Store current working directory and change to notebook directory
    # so relative paths in the notebook resolve correctly
    old_cwd = os.getcwd()
    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    os.chdir(nb_dir)
    
    # We will compile and run each code cell in the global namespace
    # to emulate Jupyter cell execution state preservation
    global_namespace = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }
    
    # Add the notebooks/ directory and its parent to path
    sys.path.insert(0, nb_dir)
    sys.path.insert(0, os.path.dirname(nb_dir))
    
    try:
        cell_idx = 0
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                cell_idx += 1
                code = "".join(cell['source'])
                # Skip empty cells
                if not code.strip():
                    continue
                
                # Exclude line-magic commands (e.g. %matplotlib inline)
                clean_lines = []
                for line in code.splitlines():
                    if line.strip().startswith('%') or line.strip().startswith('!'):
                        continue
                    clean_lines.append(line)
                clean_code = "\n".join(clean_lines)
                
                # Run the cell
                exec(clean_code, global_namespace)
        print(f"SUCCESS: Executed {cell_idx} code cells in {nb_path}")
        return True
    except Exception as e:
        print(f"ERROR: Execution failed in notebook {nb_path}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore directory state
        os.chdir(old_cwd)
        if nb_dir in sys.path:
            sys.path.remove(nb_dir)
        parent_dir = os.path.dirname(nb_dir)
        if parent_dir in sys.path:
            sys.path.remove(parent_dir)

if __name__ == '__main__':
    # Execute notebook 1
    success1 = run_notebook(os.path.join('notebooks', '01_feature_extraction.ipynb'))
    if not success1:
        sys.exit(1)
        
    # Execute notebook 2
    success2 = run_notebook(os.path.join('notebooks', '02_modeling_and_viz.ipynb'))
    if not success2:
        sys.exit(1)
        
    print("\n==================================================")
    print("ALL NOTEBOOKS EXECUTED SUCCESSFULLY!")
    print("Metrics and visualizations have been generated.")
    print("==================================================")
