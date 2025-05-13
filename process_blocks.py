import streamlit as st

class ProcessBlock:
    def __init__(self, block_id, name, block_type, parameters):
        self.block_id = block_id
        self.name = name
        self.block_type = block_type
        self.parameters = parameters

    def to_dict(self):
        return {
            "block_id": self.block_id,
            "name": self.name,
            "block_type": self.block_type,
            "parameters": self.parameters,
        }

# Función para inicializar bloques en el estado de la sesión
def initialize_process_blocks():
    if 'process_blocks' not in st.session_state:
        st.session_state.process_blocks = []

# Función para agregar un nuevo bloque
def add_process_block(name, block_type, parameters):
    block_id = len(st.session_state.process_blocks) + 1
    new_block = ProcessBlock(block_id, name, block_type, parameters)
    st.session_state.process_blocks.append(new_block)

# Función para eliminar un bloque
def remove_process_block(block_id):
    st.session_state.process_blocks = [
        block for block in st.session_state.process_blocks if block.block_id != block_id
    ]
