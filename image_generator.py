"""
Simple image generator for social sharing of Trump Tweet Challenge results.

Creates dynamic OpenGraph images with score, level, and branding.
"""

from PIL import Image, ImageDraw, ImageFont
import os
from typing import Tuple, Optional
import textwrap


class ShareImageGenerator:
    """Generate social sharing images for Trump Tweet Challenge results."""
    
    def __init__(self, output_dir: str = "frontend/images/share"):
        """
        Initialize the image generator.
        
        Args:
            output_dir: Directory to save generated images
        """
        self.output_dir = output_dir
        self.width = 1200
        self.height = 630  # Standard OpenGraph dimensions
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Try to load fonts, fall back to default if not available
        self.title_font = self._load_font(size=60, weight="bold")
        self.score_font = self._load_font(size=120, weight="bold")
        self.level_font = self._load_font(size=40, weight="normal")
        self.text_font = self._load_font(size=30, weight="normal")
    
    def _load_font(self, size: int, weight: str = "normal") -> ImageFont.FreeTypeFont:
        """Load font with fallback to default."""
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Arial.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except OSError:
                    continue
        
        # Fallback to default font
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()
    
    def create_gradient_background(self) -> Image.Image:
        """Create a gradient background similar to the website."""
        img = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(img)
        
        # Create a multi-color gradient
        for y in range(self.height):
            # Calculate color transition
            ratio = y / self.height
            
            if ratio < 0.33:
                # Blue to purple
                r = int(102 + (118 - 102) * (ratio * 3))
                g = int(126 + (75 - 126) * (ratio * 3))
                b = int(234 + (162 - 234) * (ratio * 3))
            elif ratio < 0.66:
                # Purple to pink
                local_ratio = (ratio - 0.33) * 3
                r = int(118 + (240 - 118) * local_ratio)
                g = int(75 + (147 - 75) * local_ratio)
                b = int(162 + (251 - 162) * local_ratio)
            else:
                # Pink to orange
                local_ratio = (ratio - 0.66) * 3
                r = int(240 + (245 - 240) * local_ratio)
                g = int(147 + (87 - 147) * local_ratio)
                b = int(251 + (108 - 251) * local_ratio)
            
            draw.line([(0, y), (self.width, y)], fill=(r, g, b))
        
        return img
    
    def add_text_with_outline(self, 
                            draw: ImageDraw.Draw, 
                            text: str, 
                            position: Tuple[int, int], 
                            font: ImageFont.FreeTypeFont, 
                            fill_color: str = "white", 
                            outline_color: str = "black",
                            outline_width: int = 2):
        """Add text with outline for better readability."""
        x, y = position
        
        # Draw outline
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
        
        # Draw main text
        draw.text(position, text, font=font, fill=fill_color)
    
    def generate_share_image(self, 
                           trump_score: int,
                           trump_level: str,
                           classification: str,
                           text_preview: str = None,
                           share_hash: str = None) -> str:
        """
        Generate a social sharing image.
        
        Args:
            trump_score: Score percentage (0-100)
            trump_level: Trump level string
            classification: "Trump" or "Non-Trump"
            text_preview: Optional preview of the original text
            share_hash: Unique hash for filename
            
        Returns:
            str: Path to generated image
        """
        # Create base image
        img = self.create_gradient_background()
        draw = ImageDraw.Draw(img)
        
        # Add semi-transparent overlay for better text readability
        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 80))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Title
        title = "🎯 TRUMP TWEET CHALLENGE"
        title_bbox = draw.textbbox((0, 0), title, font=self.title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.width - title_width) // 2
        self.add_text_with_outline(draw, title, (title_x, 50), self.title_font, 
                                 fill_color="#FFD700", outline_color="black", outline_width=3)
        
        # Score (main focus)
        score_text = f"{trump_score}%"
        score_bbox = draw.textbbox((0, 0), score_text, font=self.score_font)
        score_width = score_bbox[2] - score_bbox[0]
        score_x = (self.width - score_width) // 2
        
        # Score color based on value
        if trump_score >= 70:
            score_color = "#32CD32"  # Green
        elif trump_score >= 40:
            score_color = "#FFA500"  # Orange
        else:
            score_color = "#FF6B6B"  # Red
            
        self.add_text_with_outline(draw, score_text, (score_x, 180), self.score_font,
                                 fill_color=score_color, outline_color="black", outline_width=4)
        
        # Trump Level
        level_bbox = draw.textbbox((0, 0), trump_level, font=self.level_font)
        level_width = level_bbox[2] - level_bbox[0]
        level_x = (self.width - level_width) // 2
        self.add_text_with_outline(draw, trump_level, (level_x, 320), self.level_font,
                                 fill_color="white", outline_color="black", outline_width=2)
        
        # Text preview (if provided)
        if text_preview:
            # Wrap text to fit
            wrapped_text = textwrap.fill(text_preview[:100] + "..." if len(text_preview) > 100 else text_preview, 
                                       width=50)
            preview_lines = wrapped_text.split('\n')[:3]  # Max 3 lines
            
            y_offset = 400
            for line in preview_lines:
                line_bbox = draw.textbbox((0, 0), f'"{line}"', font=self.text_font)
                line_width = line_bbox[2] - line_bbox[0]
                line_x = (self.width - line_width) // 2
                self.add_text_with_outline(draw, f'"{line}"', (line_x, y_offset), self.text_font,
                                         fill_color="#E0E0E0", outline_color="black", outline_width=1)
                y_offset += 35
        
        # Footer
        footer = "Can you write like 45-47? Take the challenge!"
        footer_bbox = draw.textbbox((0, 0), footer, font=self.text_font)
        footer_width = footer_bbox[2] - footer_bbox[0]
        footer_x = (self.width - footer_width) // 2
        self.add_text_with_outline(draw, footer, (footer_x, 550), self.text_font,
                                 fill_color="#FFD700", outline_color="black", outline_width=2)
        
        # Save image
        filename = f"trump_score_{share_hash or trump_score}_{trump_level.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        img.save(filepath, 'PNG', quality=95)
        
        return filepath
    
    def generate_default_share_image(self) -> str:
        """Generate a default sharing image for the main site."""
        img = self.create_gradient_background()
        draw = ImageDraw.Draw(img)
        
        # Add semi-transparent overlay
        overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 80))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Main title
        title = "🎯 TRUMP TWEET CHALLENGE"
        title_bbox = draw.textbbox((0, 0), title, font=self.score_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (self.width - title_width) // 2
        self.add_text_with_outline(draw, title, (title_x, 150), self.score_font,
                                 fill_color="#FFD700", outline_color="black", outline_width=4)
        
        # Subtitle
        subtitle = "Can you write like 45-47?"
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=self.title_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (self.width - subtitle_width) // 2
        self.add_text_with_outline(draw, subtitle, (subtitle_x, 300), self.title_font,
                                 fill_color="white", outline_color="black", outline_width=3)
        
        # Call to action
        cta = "Test your Trump tweeting skills now!"
        cta_bbox = draw.textbbox((0, 0), cta, font=self.level_font)
        cta_width = cta_bbox[2] - cta_bbox[0]
        cta_x = (self.width - cta_width) // 2
        self.add_text_with_outline(draw, cta, (cta_x, 450), self.level_font,
                                 fill_color="#32CD32", outline_color="black", outline_width=2)
        
        # Save image
        filepath = os.path.join(self.output_dir, "default_share.png")
        img.save(filepath, 'PNG', quality=95)
        
        return filepath


# Global generator instance
image_gen = ShareImageGenerator()
