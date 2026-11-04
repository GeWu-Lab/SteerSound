#!/usr/bin/env python3
"""Fine-tune an existing editor with the standard flow-matching objective."""
from utils.trainer import main

if __name__ == '__main__':
    main('finetune', 'complex')
