# Post Like Trump Challenge - Frontend

A fun, interactive web application that challenges users to write tweets that sound like Donald Trump.

## Features

- **Interactive Trump-o-meter**: Visual score representation with fun level names
- **Real-time character counting**: Twitter-style character limit tracking
- **Example tweets**: Pre-loaded examples to inspire users
- **Responsive design**: Works on desktop and mobile
- **Fun scoring system**: From "Definitely Not Trump" to "Certified Trump"

## Trump-o-meter Levels

- 🏆 **Certified Trump** (95%+): The highest honor!
- 👔 **Donald Trump Jr.** (85%+): Executive level Trump-speak
- 🏗️ **Eric Trump** (70%+): Solid Trump vibes
- 💎 **Tiffany Trump** (55%+): Getting there!
- 🇺🇸 **Trump Supporter** (40%+): On the right track
- 🤔 **Trump Curious** (25%+): Showing some potential
- 🚫 **Definitely Not Trump** (<25%): Try again!

## Setup

1. Make sure the API is running on `http://localhost:8000`
2. Open `index.html` in a web browser
3. Start typing Trump-like tweets!

## Usage

1. Enter your tweet in the text area (280 character limit)
2. Click "Analyze My Tweet!"
3. See your Trump score and level
4. Try again to improve your score!

## API Integration

The frontend calls the following API endpoints:
- `POST /classify` - Main classification endpoint
- `GET /health` - Health check endpoint

## Browser Compatibility

- Chrome (recommended)
- Firefox
- Safari
- Edge

## Technologies Used

- HTML5
- CSS3 (with Bootstrap 5)
- Vanilla JavaScript
- Font Awesome icons
