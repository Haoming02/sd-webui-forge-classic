# Forge Canvas - Layer Management System
# AGPL V3
# Handles layer creation, composition, blending modes, and serialization

import uuid
import base64
import io
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from PIL import Image
import numpy as np


class BlendMode(Enum):
    """Supported blend modes for layers"""
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    ADD = "add"
    SUBTRACT = "subtract"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"


@dataclass
class Layer:
    """Represents a single canvas layer"""
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    name: str = "Layer"
    visible: bool = True
    opacity: float = 1.0  # 0.0 to 1.0
    blend_mode: str = BlendMode.NORMAL.value
    x_offset: int = 0
    y_offset: int = 0
    width: int = 512
    height: int = 512
    # Store layer data as base64-encoded PNG or token reference
    data: Optional[str] = None
    data_token: Optional[str] = None  # Reference to server-stored image (forge-canvas://<id>)
    locked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert layer to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'visible': self.visible,
            'opacity': self.opacity,
            'blend_mode': self.blend_mode,
            'x_offset': self.x_offset,
            'y_offset': self.y_offset,
            'width': self.width,
            'height': self.height,
            'data': self.data,
            'data_token': self.data_token,
            'locked': self.locked
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Layer':
        """Create layer from dictionary"""
        return Layer(
            id=data.get('id', uuid.uuid4().hex),
            name=data.get('name', 'Layer'),
            visible=data.get('visible', True),
            opacity=data.get('opacity', 1.0),
            blend_mode=data.get('blend_mode', BlendMode.NORMAL.value),
            x_offset=data.get('x_offset', 0),
            y_offset=data.get('y_offset', 0),
            width=data.get('width', 512),
            height=data.get('height', 512),
            data=data.get('data'),
            data_token=data.get('data_token'),
            locked=data.get('locked', False)
        )


class LayerManager:
    """Manages multiple layers and composition operations"""
    
    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self.layers: List[Layer] = []
        self.active_layer_id: Optional[str] = None
        
        # Create default layer
        self.add_layer("Background")
    
    def add_layer(self, name: str = "Layer", index: Optional[int] = None) -> Layer:
        """Add a new layer to the layer stack
        
        Args:
            name: Name of the layer
            index: Position in stack (None = top of stack)
        
        Returns:
            The created Layer object
        """
        layer = Layer(
            name=name,
            width=self.width,
            height=self.height
        )
        
        if index is None:
            self.layers.append(layer)
        else:
            self.layers.insert(index, layer)
        
        # Set as active layer if none selected
        if self.active_layer_id is None:
            self.active_layer_id = layer.id
        
        return layer
    
    def delete_layer(self, layer_id: str) -> bool:
        """Delete a layer by ID
        
        Args:
            layer_id: ID of layer to delete
        
        Returns:
            True if deleted successfully, False otherwise
        """
        layer = self.get_layer(layer_id)
        if not layer or len(self.layers) <= 1:
            return False
        
        self.layers = [l for l in self.layers if l.id != layer_id]
        
        # Update active layer if deleted
        if self.active_layer_id == layer_id:
            self.active_layer_id = self.layers[0].id if self.layers else None
        
        return True
    
    def get_layer(self, layer_id: str) -> Optional[Layer]:
        """Get a layer by ID"""
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None
    
    def get_active_layer(self) -> Optional[Layer]:
        """Get the currently active layer"""
        if self.active_layer_id:
            return self.get_layer(self.active_layer_id)
        return self.layers[0] if self.layers else None
    
    def set_active_layer(self, layer_id: str) -> bool:
        """Set the active layer
        
        Returns:
            True if layer exists, False otherwise
        """
        if self.get_layer(layer_id):
            self.active_layer_id = layer_id
            return True
        return False
    
    def reorder_layer(self, layer_id: str, new_index: int) -> bool:
        """Move a layer to a new position in the stack
        
        Args:
            layer_id: ID of layer to move
            new_index: New position (0 = bottom, len-1 = top)
        
        Returns:
            True if reordered successfully, False otherwise
        """
        layer = self.get_layer(layer_id)
        if not layer:
            return False
        
        current_index = self.layers.index(layer)
        self.layers.pop(current_index)
        self.layers.insert(new_index, layer)
        return True
    
    def merge_down(self, layer_id: str) -> bool:
        """Merge a layer with the layer below it
        
        Args:
            layer_id: ID of layer to merge down
        
        Returns:
            True if merge successful, False otherwise
        """
        layer = self.get_layer(layer_id)
        if not layer:
            return False
        
        index = self.layers.index(layer)
        if index == 0:
            return False  # Can't merge bottom layer
        
        bottom_layer = self.layers[index - 1]
        
        # Composite this layer onto the one below
        try:
            composite = self._composite_two_layers(bottom_layer, layer)
            if composite:
                bottom_layer.data = composite
                bottom_layer.data_token = None
            
            # Delete this layer
            self.layers.pop(index)
            
            # Update active layer if needed
            if self.active_layer_id == layer_id:
                self.active_layer_id = bottom_layer.id
            
            return True
        except Exception as e:
            print(f"Error merging layers: {e}")
            return False
    
    def flatten_image(self) -> Optional[Tuple[np.ndarray, 'Image.Image']]:
        """Flatten all layers to a single image
        
        Returns:
            Tuple of (numpy array, PIL Image) or None on error
        """
        try:
            # Create base image with all visible layers composited
            base = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            
            for layer in self.layers:
                if not layer.visible:
                    continue
                
                if layer.data_token:
                    # Would need to fetch from server in actual implementation
                    # For now, skip server-stored layers
                    continue
                
                if layer.data:
                    try:
                        img = self._load_layer_image(layer)
                        if img:
                            self._composite_onto_base(base, img, layer)
                    except Exception as e:
                        print(f"Error compositing layer {layer.name}: {e}")
                        continue
            
            # Convert to numpy array
            array = np.array(base)
            return array, base
        except Exception as e:
            print(f"Error flattening image: {e}")
            return None
    
    def set_layer_opacity(self, layer_id: str, opacity: float) -> bool:
        """Set layer opacity (0.0 to 1.0)"""
        layer = self.get_layer(layer_id)
        if layer:
            layer.opacity = max(0.0, min(1.0, opacity))
            return True
        return False
    
    def set_layer_visibility(self, layer_id: str, visible: bool) -> bool:
        """Set layer visibility"""
        layer = self.get_layer(layer_id)
        if layer:
            layer.visible = visible
            return True
        return False
    
    def set_layer_blend_mode(self, layer_id: str, blend_mode: str) -> bool:
        """Set layer blend mode"""
        layer = self.get_layer(layer_id)
        if layer and blend_mode in [bm.value for bm in BlendMode]:
            layer.blend_mode = blend_mode
            return True
        return False
    
    def rename_layer(self, layer_id: str, new_name: str) -> bool:
        """Rename a layer"""
        layer = self.get_layer(layer_id)
        if layer:
            layer.name = new_name
            return True
        return False
    
    def lock_layer(self, layer_id: str, locked: bool) -> bool:
        """Lock/unlock a layer"""
        layer = self.get_layer(layer_id)
        if layer:
            layer.locked = locked
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize layer manager state"""
        return {
            'width': self.width,
            'height': self.height,
            'active_layer_id': self.active_layer_id,
            'layers': [layer.to_dict() for layer in self.layers]
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Deserialize layer manager state"""
        self.width = data.get('width', self.width)
        self.height = data.get('height', self.height)
        self.active_layer_id = data.get('active_layer_id')
        self.layers = [Layer.from_dict(l) for l in data.get('layers', [])]
    
    # Private helper methods
    
    def _load_layer_image(self, layer: Layer) -> Optional[Image.Image]:
        """Load layer image from base64 data"""
        if not layer.data:
            return None
        
        try:
            if layer.data.startswith('data:image/png;base64,'):
                base64_str = layer.data.replace('data:image/png;base64,', '')
            else:
                base64_str = layer.data
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return image.convert('RGBA')
        except Exception as e:
            print(f"Error loading layer image: {e}")
            return None
    
    def _composite_onto_base(self, base: Image.Image, layer_img: Image.Image, layer: Layer) -> None:
        """Composite a layer image onto the base image with blending"""
        if not layer.visible or layer.opacity == 0:
            return
        
        # Resize layer image to match canvas if needed
        if layer_img.size != (self.width, self.height):
            layer_img = layer_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        # Apply opacity
        if layer.opacity < 1.0:
            if layer_img.mode != 'RGBA':
                layer_img = layer_img.convert('RGBA')
            alpha = layer_img.split()[3]
            alpha = Image.new('L', alpha.size, int(255 * layer.opacity))
            layer_img.putalpha(alpha)
        
        # Apply blend mode
        if layer.blend_mode == BlendMode.NORMAL.value:
            base.paste(layer_img, (layer.x_offset, layer.y_offset), layer_img)
        else:
            # Fallback: use normal blend mode for now
            # Full blend mode implementation would require more complex math
            base.paste(layer_img, (layer.x_offset, layer.y_offset), layer_img)
    
    def _composite_two_layers(self, bottom: Layer, top: Layer) -> Optional[str]:
        """Composite two layers and return base64-encoded result"""
        try:
            bottom_img = self._load_layer_image(bottom)
            top_img = self._load_layer_image(top)
            
            if not bottom_img or not top_img:
                return None
            
            # Create result image
            result = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            
            # Paste bottom layer
            if bottom_img.size != (self.width, self.height):
                bottom_img = bottom_img.resize((self.width, self.height), Image.Resampling.LANCZOS)
            result.paste(bottom_img, (0, 0), bottom_img)
            
            # Paste top layer with blending
            self._composite_onto_base(result, top_img, top)
            
            # Convert to base64
            buffered = io.BytesIO()
            result.save(buffered, format='PNG')
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error compositing layers: {e}")
            return None
