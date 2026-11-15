# What changed in xLamina

The newest version first. In the program: Help > What's new.

## 1.1.0 (15 November 2026)

- **Refine** works on any output of the last render (B, C, CBn …), not only B. Its row is always shown and says what it needs.
- **Clone brush** in Retouch: copies from another place of the image, matched in colour and brightness, for dust specks. Press C, Alt + click the place to copy from, paint.
- **Make composite:** the methods are chosen under Methods (B, C and CBn to begin with). An image of another size, such as another program's stack of the same frames, is lined up first.
- **Compare window:** each toolbar row has a colour, strong while its mode is on. Hide frames takes the stack's frames out of the list. A saved retouch keeps the field of view in its name.
- While retouching, picking another image for the left pane moves the retouch to it. A tool that cannot do what was asked says why beside the cursor.
- Fixed: a brush stroke stayed open when a tool's shortcut was pressed during the drag.
- **Help > What's new** shows this list.

## 1.0.16 (14 November 2026)

- **Compare window** with 2, 3 or 4 panes; retouch from any pane.
- The stack's frames in the list, lined up with the outputs. **Frame here** finds the frame that is sharpest at a spot.
- **Refine:** marked regions of B rendered again with more weak detail.
- **Make composite:** finished images stacked again as if they were frames.
- Align and Align here for an image of another crop or scale; units for measured lines and the scale bar.
- B on stacks of fewer than 192 frames: fewer blobs on flat crystal faces.

## 1.0.13 to 1.0.15 (14 November 2026)

- Depth map (developer controls): a click holds a layer; the arrow keys step through the layers. Also in the compare window.
- Measure in the compare window.
- Open… lists developed images only; raw files are a choice in the file type box.

## 1.0.10 to 1.0.12 (14 November 2026)

- A scale bar in the viewers, and Measure (M).
- Open… shows the photos: one photo opens its folder, several make a stack of just those.
- Dust map: a window shows what is found; the C outputs are corrected too.
- Live view: Full, at the screen's real pixels.

## 1.0.8 and 1.0.9 (14 November 2026)

- **Dust map:** File > Set dust map takes a photo of a white background and removes the sensor's dust from every frame.
- A stack of chosen files; a folder of more than 500 images asks first.

## 1.0.7 (14 November 2026)

- Runs on more PCs: its own GPU code for RTX 30, 40 and 50 cards, less memory on small cards.
- A failed render says what happened and what to try. A session log; Help > Report a problem.
- The manual: Help > Manual (F1).

## 1.0.6 (14 November 2026)

- The depth map in colour (developer controls), with the frame under the cursor.
- Stacks too large for the graphics cards' memory render in strips.

## 1.0.2 to 1.0.5 (11 November 2026)

- Fixed in 1.0.2: at each start, earlier versions deleted files with `.old` in their name in the program's folder.
- Folders and files with names in any script open and render.
- Updates are taken only from this project's releases, and checked.
- The zip holds one folder, xLamina.

## 1.0.1 (11 November 2026)

- Compressed TIFF frames (LZW, ZIP, PackBits).
- Batch render: several folders one after the other.
- A PC without a usable NVIDIA card is told so. Damaged TIFF files are refused.

## 1.0.0 (9 November 2026)

- GPU alignment with the rolling-shutter correction, the depth-map output B, the pyramid output C, a geometry report, and the window with its live view.
