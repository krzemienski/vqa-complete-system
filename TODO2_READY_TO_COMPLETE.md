# TODO 2: Ready to Complete

## Current Status
- ✅ Scripts created and tested
- ✅ Test download successful
- ✅ Test degradation successful  
- ✅ Directory structure ready
- ✅ Dependencies verified (ffmpeg installed)

## To Complete TODO 2

### Option 1: Run Everything at Once
```bash
cd scripts
./complete_todo2.sh
```

This will:
1. Download all test videos (~500MB-1GB)
2. Create all degraded versions (~2-3GB)
3. Generate metadata file
4. Show summary

Estimated time: 25-55 minutes

### Option 2: Run Separately
```bash
cd scripts

# Step 1: Download originals
python download_test_videos.py

# Step 2: Create degradations  
python create_degraded_videos.py
```

## Expected Results

### Original Videos (7 files)
- Big Buck Bunny: 1080p, 720p, 480p
- Tears of Steel: 1080p, 720p
- Sintel: 1080p, 720p

### Degraded Videos (126 files)
- 7 originals × 18 degradation types
- Categories: compression, noise, blur, scaling, temporal, color, combined
- Metadata file with all degradation details

## Disk Space Requirements
- Original videos: ~500MB-1GB
- Degraded videos: ~2-3GB
- Total needed: ~4GB free space

## Verification
After completion, verify with:
```bash
# Count videos
find ../test_videos -name "*.mp4" | wc -l
# Should show 133 total (7 original + 126 degraded)

# Check metadata
cat ../test_videos/degraded/degradation_metadata.json
```

## Current Test Files
- ✅ `test_blazes.mp4` - Successfully downloaded
- ✅ `test_blazes_crf40.mp4` - Successfully degraded

## Ready to Proceed
All systems tested and ready. Run `./complete_todo2.sh` when ready to download and process all videos.