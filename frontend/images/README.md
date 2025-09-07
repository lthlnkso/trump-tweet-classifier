# Images Folder

This folder is for storing images that can be used in the frontend application.

## Suggested Images to Add:

### Backgrounds:
- `trump-background.jpg` - Optional background image for the main page
- `american-flag.jpg` - Patriotic background option

### Icons/Graphics:
- `trump-avatar.png` - Profile-style image for high scores
- `podium.png` - For "Certified Trump" achievements
- `gold-star.png` - Achievement badges

### UI Elements:
- `twitter-logo.png` - Social media theme
- `checkmark.png` - Success indicators

## How to Use:

1. Add images to this folder
2. Reference them in HTML using: `/static/images/filename.jpg`
3. Or in CSS using: `url('/static/images/filename.jpg')`

## Recommended Sizes:

- Background images: 1920x1080px or similar
- Icons: 64x64px to 256x256px  
- Avatars: 150x150px (circular)

## File Formats:

- **PNG**: For icons, transparent backgrounds
- **JPG**: For photos, backgrounds
- **SVG**: For scalable icons
- **WebP**: For modern browsers (smaller file sizes)

## Examples:

```html
<!-- In HTML -->
<img src="/static/images/trump-avatar.png" alt="Trump Avatar">

<!-- In CSS -->
.background {
    background-image: url('/static/images/american-flag.jpg');
}
```
