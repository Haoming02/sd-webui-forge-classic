# Canvas Enhancements for Inpaint Sketch Tool

This document describes the enhancements made to the Forge Canvas inpaint sketch tool, located in `modules_forge/forge_canvas/`.

## Overview

Three major features have been added to improve the drawing experience in the inpaint sketch tool:

1. **Brush Shape Selection** (Circle/Rectangle)
2. **Straight Line Drawing** (Ctrl+Click)
3. **Brush Rotation** (0-360°)

---

## Feature 1: Brush Shape Toggle

### Description
Added the ability to switch between circular and rectangular brush shapes.

### Implementation
- **Button**: New brush shape button (○/□) in toolbar
- **Keyboard Shortcut**: Press `B` to toggle between shapes
- **Visual Indicator**: Cursor preview changes shape to match selected brush

### Modified Files
- `canvas.html`: Added brush shape button (line 36)
- `canvas.js`:
  - Added `brushShape` state variable (line 101)
  - Implemented `toggleBrushShape()` method (lines 994-1010)
  - Updated `handleDraw()` to apply correct line cap (lines 590-595)
  - Event listeners for button and keyboard shortcut (lines 240-243, 498-501)
- `canvas.css`: No changes required (uses existing button styles)

### Usage
1. Click the brush shape button (shows ○ for circle, □ for rectangle)
2. Or press `B` key to toggle
3. Cursor preview updates to show current shape
4. Draw with selected brush shape

---

## Feature 2: Straight Line Drawing

### Description
Hold Ctrl and click to draw perfectly straight lines from the last drawn point.

### Implementation
- **Keyboard Modifier**: Ctrl+Click
- **Behavior**: Draws straight line from last point to click location
- **Visual Hint**: Added "Hold Ctrl + Click for straight line" text in toolbar
- **Persistence**: Last point is remembered across strokes

### Modified Files
- `canvas.html`: Added hint text in toolbar (line 31)
- `canvas.js`:
  - Added `lastLinePoint` tracking (line 102)
  - Implemented Ctrl+Click detection in pointerdown handler (lines 296-304)
  - Updates last point after each stroke (lines 310-327)
  - Resets on canvas clear (lines 738, 751)
- `canvas.css`:
  - Added `.forge-toolbar-hint` styling (lines 95-101)
  - Updated `.forge-toolbar-box-a` for flexbox layout (lines 88-93)

### Usage
1. Draw a stroke normally
2. Hold Ctrl and click anywhere to draw a straight line from the end of your last stroke
3. Continue Ctrl+clicking to chain multiple straight lines
4. Release Ctrl to return to freehand drawing

---

## Feature 3: Brush Rotation

### Description
Rotate the rectangular brush 0-360 degrees for angled strokes. Particularly useful for architectural drawing or precise masking.

### Implementation
- **Slider**: New rotation slider (0-360°) in toolbar
- **Keyboard Shortcut**: Hold `D` + scroll wheel to adjust rotation
- **Visual Indicator**: Cursor preview rotates to match brush angle
- **Smart Rendering**:
  - At 0° rotation: Uses fast native line rendering
  - At >0° rotation: Switches to optimized stamp-based rendering
- **Works with all brush settings**: Opacity, softness, width all function correctly

### Modified Files
- `canvas.html`: Added rotation slider UI (lines 55-60)
- `canvas.js`:
  - Added `scribbleRotation` and `scribbleRotationFixed` parameters (lines 49-50, 81-82)
  - Added `_held_D` state for keyboard shortcut (line 100)
  - Implemented rotation slider and event handlers (lines 156-161, 284-289, 408-413)
  - Implemented `drawWithRotatedStamps()` method for rotated rendering (lines 644-689)
  - Implemented `drawRotatedRectangleStamp()` for individual stamps (lines 691-718)
  - Updated cursor indicator rotation (multiple locations)
  - Keyboard shortcut handlers (lines 529, 536)
- `canvas.py`:
  - Added rotation parameters to constructor (line 125)
  - Updated JavaScript instantiation call (line 138)
- `canvas.css`: No changes required

### Technical Details

**Rendering Strategy:**
- **Circle brush**: Rotation has no visual effect (as expected), but slider remains available for future brush shapes
- **Rectangle at 0°**: Uses canvas 2D context's native `stroke()` with `lineCap="square"`
- **Rectangle at >0°**: Switches to stamp-based rendering:
  - Draws individual rotated rectangles along the stroke path
  - Interpolates between points with 10% spacing for smooth coverage
  - Handles softness by layering multiple sized rectangles with alpha blending
  - Properly supports eraser mode (alpha = 0) and all opacity levels

**Performance:**
- Adaptive stamp spacing prevents gaps while maintaining performance
- Only activates stamp rendering when rotation > 0°
- Efficient canvas transform operations (save/translate/rotate/restore)

### Usage
1. Select rectangle brush (press `B` or click brush button)
2. Adjust rotation slider or hold `D` and scroll
3. Cursor preview shows rotation angle
4. Draw with rotated rectangular brush
5. Works seamlessly with width, opacity, and softness settings

---

## Benefits

### User Experience Improvements
- **More precise masking**: Straight lines and rotated brushes enable cleaner edges
- **Faster workflow**: Ctrl+click is much faster than trying to draw straight lines by hand
- **Professional results**: Rectangle rotation enables architectural/geometric masking
- **Intuitive controls**: Clear visual feedback and keyboard shortcuts

### Technical Quality
- **Backward compatible**: All changes are additive, existing functionality unchanged
- **No performance impact**: Smart rendering only activates when needed
- **Clean code**: Well-documented, follows existing patterns
- **Extensible**: Architecture supports future brush shapes (triangle, etc.)

---

## Testing Recommendations

When reviewing these changes, test:

1. **Brush Shape Toggle**:
   - Switch between circle and rectangle
   - Verify cursor indicator updates
   - Test with various brush widths

2. **Straight Lines**:
   - Draw a curve, then Ctrl+click to extend with straight line
   - Chain multiple straight lines together
   - Verify reset after clearing canvas

3. **Rotation**:
   - Test rotation at 0°, 45°, 90°, 180° with rectangle brush
   - Verify smooth rendering with softness enabled
   - Test eraser mode (opacity = 0) with rotation
   - Confirm circle brush ignores rotation (as expected)

4. **Combined Features**:
   - Use rotated rectangle with Ctrl+click straight lines
   - Switch brush shapes while rotation is set
   - Test all combinations of width/opacity/softness with rotation

---

## Future Enhancements

The architecture now supports:
- **Triangle brush**: Stamp-based system is ready for custom shapes
- **Custom brush patterns**: Could load image-based brush shapes
- **More rotation shortcuts**: Could add preset angles (0°, 45°, 90°)
- **Rotation lock**: Option to lock rotation to 45° increments

---

## Files Modified

- `modules_forge/forge_canvas/canvas.html`
- `modules_forge/forge_canvas/canvas.js`
- `modules_forge/forge_canvas/canvas.py`
- `modules_forge/forge_canvas/canvas.css`

## Compatibility

- ✅ No breaking changes
- ✅ Backward compatible with existing code
- ✅ Works with all existing canvas features
- ✅ No new dependencies

---

**Author**: SillySilk
**Date**: 2025-01-21
**Based on**: Forge Canvas by lllyasviel
