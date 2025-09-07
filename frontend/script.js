// API Configuration - use relative URLs since we're served from the same domain
const API_BASE_URL = '';

// Character counter
document.getElementById('tweetText').addEventListener('input', function() {
    const charCount = this.value.length;
    document.getElementById('charCount').textContent = charCount;
    
    // Change color based on character count
    const charCountElement = document.getElementById('charCount');
    if (charCount > 550) {
        charCountElement.style.color = '#e74c3c';
    } else if (charCount > 450) {
        charCountElement.style.color = '#f39c12';
    } else {
        charCountElement.style.color = '#27ae60';
    }
});

// Form submission
document.getElementById('trumpForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const tweetText = document.getElementById('tweetText').value.trim();
    if (!tweetText) {
        alert('Please enter some text!');
        return;
    }
    
    // Show loading state
    showLoading(true);
    
    try {
        const response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                text: tweetText,
                session_id: generateSessionId()
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResult(result);
        
    } catch (error) {
        console.error('Error:', error);
        alert('Sorry! There was an error analyzing your tweet. Make sure the API is running on localhost:8000');
    } finally {
        showLoading(false);
    }
});

function showLoading(isLoading) {
    const submitText = document.querySelector('.submit-text');
    const loadingSpinner = document.querySelector('.loading-spinner');
    const submitBtn = document.querySelector('.submit-btn');
    
    if (isLoading) {
        submitText.style.display = 'none';
        loadingSpinner.style.display = 'inline';
        submitBtn.disabled = true;
    } else {
        submitText.style.display = 'inline';
        loadingSpinner.style.display = 'none';
        submitBtn.disabled = false;
    }
}

function displayResult(result) {
    const trumpScore = result.trump_score;
    
    // Store submission ID for feedback
    currentSubmissionId = result.submission_id || null;
    
    // Reset animations and feedback
    resetModalAnimations();
    resetFeedbackSection();
    
    // Get the user's tweet text
    const userText = document.getElementById('tweetText').value;
    
    // Update tweet display in modal
    document.getElementById('userTweetText').textContent = userText;
    document.getElementById('tweetLength').textContent = `${userText.length} characters`;
    
    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('resultsModal'));
    modal.show();
    
    // Animate score counter
    setTimeout(() => {
        animateScore(trumpScore);
    }, 500);
    
    // Update Trump level with styling and animation
    setTimeout(() => {
        const trumpLevelBig = document.getElementById('trumpLevelBig');
        trumpLevelBig.textContent = result.trump_level;
        trumpLevelBig.className = 'trump-level-big trump-level-compact ' + getTrumpLevelClass(result.trump_level);
        
        // Show appropriate level image based on score
        showLevelImage(trumpScore, result.trump_level);
    }, 1000);
    
    // Animate modern meter
    setTimeout(() => {
        const meterFillModern = document.getElementById('meterFillModern');
        meterFillModern.style.width = `${trumpScore}%`;
    }, 1500);
    
    // Update details
    setTimeout(() => {
        document.getElementById('predictionModal').textContent = result.prediction;
        document.getElementById('confidenceModal').textContent = `${(result.confidence * 100).toFixed(1)}%`;
    }, 2000);
    
    // Celebration for high scores
    if (trumpScore >= 85) {
        setTimeout(() => {
            celebrateHighScore();
        }, 2500);
    }
}

function resetModalAnimations() {
    // Reset score
    document.getElementById('scoreNumber').textContent = '0';
    document.getElementById('meterFillModern').style.width = '0%';
    document.getElementById('trumpLevelBig').textContent = '';
    document.getElementById('predictionModal').textContent = '';
    document.getElementById('confidenceModal').textContent = '';
    
    // Reset tweet display
    document.getElementById('userTweetText').textContent = '';
    document.getElementById('tweetLength').textContent = '';
    
    // Hide all level images
    document.getElementById('certifiedTrumpImage').style.display = 'none';
    document.getElementById('donJrImage').style.display = 'none';
    document.getElementById('ericTrumpImage').style.display = 'none';
    document.getElementById('tiffanyTrumpImage').style.display = 'none';
    document.getElementById('trumpSupporterImage').style.display = 'none';
    document.getElementById('trumpThumbsUpImage').style.display = 'none';
    document.getElementById('trumpDisappointedImage').style.display = 'none';
}

function animateScore(targetScore) {
    const scoreElement = document.getElementById('scoreNumber');
    let currentScore = 0;
    const increment = Math.ceil(targetScore / 50); // 50 steps
    const interval = 40; // milliseconds
    
    const timer = setInterval(() => {
        currentScore += increment;
        if (currentScore >= targetScore) {
            currentScore = targetScore;
            clearInterval(timer);
        }
        scoreElement.textContent = currentScore;
    }, interval);
}

function getTrumpLevelClass(level) {
    if (level.includes('Certified Trump')) return 'certified';
    if (level.includes('Donald Trump Jr.')) return 'don-jr';
    if (level.includes('Eric Trump')) return 'eric';
    if (level.includes('Tiffany Trump')) return 'tiffany';
    if (level.includes('Trump Supporter')) return 'supporter';
    if (level.includes('Trump Curious')) return 'curious';
    return 'not-trump';
}

function showLevelImage(score, level) {
    // Show appropriate image based on Trump level
    if (level.includes('Certified Trump')) {
        document.getElementById('certifiedTrumpImage').style.display = 'block';
        // Also show thumbs up for ultimate achievement
        setTimeout(() => {
            document.getElementById('trumpThumbsUpImage').style.display = 'block';
        }, 500);
    } else if (level.includes('Donald Trump Jr.')) {
        document.getElementById('donJrImage').style.display = 'block';
    } else if (level.includes('Eric Trump')) {
        document.getElementById('ericTrumpImage').style.display = 'block';
    } else if (level.includes('Tiffany Trump')) {
        document.getElementById('tiffanyTrumpImage').style.display = 'block';
    } else if (level.includes('Trump Supporter')) {
        document.getElementById('trumpSupporterImage').style.display = 'block';
    } else if (level.includes('Trump Curious')) {
        // Show disappointed Trump for low scores
        document.getElementById('trumpDisappointedImage').style.display = 'block';
    } else {
        // "Definitely Not Trump" - show disappointed
        document.getElementById('trumpDisappointedImage').style.display = 'block';
    }
    
    // Show thumbs up for any score above 70% (additional celebration)
    if (score >= 70 && !level.includes('Certified Trump')) {
        setTimeout(() => {
            document.getElementById('trumpThumbsUpImage').style.display = 'block';
        }, 800);
    }
}

function celebrateHighScore() {
    // Celebration animation for high scores
    const modal = document.querySelector('.results-modal');
    modal.classList.add('celebration');
    
    // Remove celebration class after animation
    setTimeout(() => {
        modal.classList.remove('celebration');
    }, 2000);
}

function useExample(exampleText) {
    document.getElementById('tweetText').value = exampleText;
    
    // Update character count
    const event = new Event('input');
    document.getElementById('tweetText').dispatchEvent(event);
    
    // Focus on textarea
    document.getElementById('tweetText').focus();
}

function resetForm() {
    document.getElementById('trumpForm').reset();
    document.getElementById('charCount').textContent = '0';
    document.getElementById('charCount').style.color = '#27ae60';
    
    // Close modal if open
    const modal = bootstrap.Modal.getInstance(document.getElementById('resultsModal'));
    if (modal) {
        modal.hide();
    }
    
    // Reset animations and feedback
    resetModalAnimations();
    resetFeedbackSection();
    
    // Focus on textarea
    document.getElementById('tweetText').focus();
}

// Health check on page load
window.addEventListener('load', async function() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            console.warn('API health check failed');
        }
    } catch (error) {
        console.warn('Cannot connect to API. Make sure it\'s running on localhost:8000');
    }
});

// Add floating sparkles effect
function createSparkle() {
    const sparkle = document.createElement('div');
    sparkle.className = 'sparkle';
    sparkle.style.left = Math.random() * 100 + 'vw';
    sparkle.style.animationDelay = Math.random() * 2 + 's';
    document.body.appendChild(sparkle);
    
    // Remove sparkle after animation
    setTimeout(() => {
        if (sparkle.parentNode) {
            sparkle.parentNode.removeChild(sparkle);
        }
    }, 4000);
}

// Create sparkles periodically
setInterval(createSparkle, 800);

// ===== FEEDBACK FUNCTIONALITY =====

let currentSubmissionId = null;
let currentFeedbackChoice = null;
let currentShareUrl = null;

function submitFeedback(agreesWithRating) {
    currentFeedbackChoice = agreesWithRating;
    
    // Hide feedback buttons and show message section
    document.querySelector('.feedback-buttons').style.display = 'none';
    document.getElementById('feedbackMessageSection').style.display = 'block';
    
    // Focus on textarea for better UX
    document.getElementById('feedbackMessage').focus();
}

function submitFeedbackWithMessage() {
    const message = document.getElementById('feedbackMessage').value.trim();
    
    // Disable button to prevent double submission
    const submitBtn = document.querySelector('.feedback-submit-btn');
    const skipBtn = document.querySelector('.feedback-skip-btn');
    submitBtn.disabled = true;
    skipBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';
    
    sendFeedbackToAPI(currentFeedbackChoice, message || null);
}

function skipFeedbackMessage() {
    sendFeedbackToAPI(currentFeedbackChoice, null);
}

function sendFeedbackToAPI(agreesWithRating, feedbackMessage) {
    const feedbackData = {
        submission_id: currentSubmissionId,
        agrees_with_rating: agreesWithRating,
        feedback_message: feedbackMessage,
        session_id: generateSessionId()
    };
    
    fetch('/feedback', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(feedbackData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showFeedbackSuccess();
        } else {
            showFeedbackError('Failed to submit feedback');
        }
    })
    .catch(error => {
        console.error('Error submitting feedback:', error);
        showFeedbackError('Network error occurred');
    });
}

function showFeedbackSuccess() {
    // Hide message section and show success
    document.getElementById('feedbackMessageSection').style.display = 'none';
    document.getElementById('feedbackSuccess').style.display = 'block';
    
    // Optional: Hide feedback section entirely after a delay
    setTimeout(() => {
        document.getElementById('feedbackSection').style.opacity = '0.7';
    }, 2000);
}

function showFeedbackError(message) {
    // Re-enable buttons
    const submitBtn = document.querySelector('.feedback-submit-btn');
    const skipBtn = document.querySelector('.feedback-skip-btn');
    submitBtn.disabled = false;
    skipBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Feedback';
    
    // Show error (you might want to add an error display element)
    console.error('Feedback error:', message);
    alert('Sorry, there was an error submitting your feedback. Please try again.');
}

function resetFeedbackSection() {
    // Reset all feedback UI elements
    document.querySelector('.feedback-buttons').style.display = 'block';
    document.getElementById('feedbackMessageSection').style.display = 'none';
    document.getElementById('feedbackSuccess').style.display = 'none';
    document.getElementById('feedbackSection').style.opacity = '1';
    document.getElementById('feedbackMessage').value = '';
    
    // Reset button states
    const submitBtn = document.querySelector('.feedback-submit-btn');
    const skipBtn = document.querySelector('.feedback-skip-btn');
    if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Feedback';
    }
    if (skipBtn) {
        skipBtn.disabled = false;
    }
    
    // Hide copy link button and reset share state
    const copyBtn = document.querySelector('.copy-link-btn');
    if (copyBtn) {
        copyBtn.style.display = 'none';
        copyBtn.disabled = false;
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy Link';
        copyBtn.style.background = 'linear-gradient(135deg, #17a2b8, #007bff)';
    }
    
    currentFeedbackChoice = null;
    currentShareUrl = null;
}

function generateSessionId() {
    // Simple session ID generation (you might want to persist this in localStorage)
    if (!window.sessionId) {
        window.sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    return window.sessionId;
}

// ===== SHARE FUNCTIONALITY =====

function shareResult() {
    if (!currentSubmissionId) {
        alert('No result to share! Please submit a tweet first.');
        return;
    }
    
    const shareBtn = document.querySelector('.share-result-btn');
    shareBtn.disabled = true;
    shareBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Share Link...';
    
    fetch('/share', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            submission_id: currentSubmissionId
        })
    })
    .then(response => response.json())
    .then(data => {
        console.log('Share API response:', data); // Debug log
        if (data.success) {
            // Always construct the URL from current window location to ensure it's correct
            const currentBase = `${window.location.protocol}//${window.location.host}`;
            const shareHash = data.share_hash;
            let shareUrl = `${currentBase}/share/${shareHash}`;
            
            // Log both URLs for comparison
            console.log('Original API URL:', data.share_url);
            console.log('Constructed URL:', shareUrl);
            
            // Store the share URL globally
            currentShareUrl = shareUrl;
            
            console.log('Share URL created:', shareUrl); // Debug log
            
            // Show the copy link button
            const copyBtn = document.querySelector('.copy-link-btn');
            copyBtn.style.display = 'inline-block';
            
            // Update button to show share created
            shareBtn.innerHTML = '<i class="fas fa-check"></i> Share Link Created!';
            shareBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            
            // Show share URL in a text field for easy copying
            showShareUrlField(shareUrl);
            
            // Reset button after delay
            setTimeout(() => {
                shareBtn.disabled = false;
                shareBtn.innerHTML = '<i class="fas fa-share-alt"></i> Share My Score!';
                shareBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
            }, 4000);
        } else {
            throw new Error('Failed to create share link');
        }
    })
    .catch(error => {
        console.error('Error creating share link:', error);
        shareBtn.disabled = false;
        shareBtn.innerHTML = '<i class="fas fa-share-alt"></i> Share My Score!';
        alert('Sorry, there was an error creating your share link. Please try again.');
    });
}

function copyToClipboard(text) {
    console.log('Attempting to copy to clipboard:', text); // Debug log
    
    return new Promise((resolve, reject) => {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text)
                .then(() => {
                    console.log('Clipboard API copy successful'); // Debug log
                    resolve(true);
                })
                .catch(err => {
                    console.warn('Clipboard API failed, trying fallback:', err);
                    // Try fallback
                    if (fallbackCopyTextToClipboard(text)) {
                        resolve(true);
                    } else {
                        reject(err);
                    }
                });
        } else {
            console.log('Using fallback copy method'); // Debug log
            // Use fallback for older browsers or non-secure contexts
            if (fallbackCopyTextToClipboard(text)) {
                resolve(true);
            } else {
                reject(new Error('Copy to clipboard failed'));
            }
        }
    });
}

function fallbackCopyTextToClipboard(text) {
    try {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.left = "-999999px";
        textArea.style.top = "-999999px";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        const successful = document.execCommand('copy');
        document.body.removeChild(textArea);
        return successful;
    } catch (err) {
        console.error('Fallback copy failed:', err);
        return false;
    }
}

function showShareUrlField(shareUrl) {
    // Create a prominent share URL display
    const shareUrlDisplay = document.createElement('div');
    shareUrlDisplay.className = 'share-url-display';
    shareUrlDisplay.innerHTML = `
        <div style="
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.95);
            color: white;
            padding: 25px;
            border-radius: 15px;
            z-index: 10000;
            max-width: 90vw;
            min-width: 400px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
            border: 3px solid rgba(255,215,0,0.5);
        ">
            <div style="text-align: center; margin-bottom: 20px;">
                <h4 style="color: #ffd700; margin: 0 0 10px 0;">
                    <i class="fas fa-share-alt"></i> Your Share Link is Ready!
                </h4>
                <p style="margin: 0; color: #ccc; font-size: 0.9em;">
                    Copy this link to share your Trump score:
                </p>
            </div>
            
            <div style="margin-bottom: 20px;">
                <input 
                    type="text" 
                    id="shareUrlInput" 
                    value="${shareUrl}" 
                    readonly
                    style="
                        width: 100%;
                        padding: 12px;
                        font-size: 14px;
                        border: 2px solid #ffd700;
                        border-radius: 8px;
                        background: white;
                        color: #333;
                        font-family: monospace;
                        box-sizing: border-box;
                    "
                >
            </div>
            
            <div style="display: flex; gap: 10px; justify-content: center; margin-bottom: 15px;">
                <button 
                    onclick="copyShareUrlFromField()" 
                    style="
                        background: linear-gradient(135deg, #007bff, #0056b3);
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 8px;
                        font-weight: bold;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    "
                    onmouseover="this.style.background='linear-gradient(135deg, #0056b3, #004085)'"
                    onmouseout="this.style.background='linear-gradient(135deg, #007bff, #0056b3)'"
                >
                    <i class="fas fa-copy"></i> Copy Link
                </button>
                
                <button 
                    onclick="selectAllShareUrl()" 
                    style="
                        background: linear-gradient(135deg, #28a745, #1e7e34);
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 8px;
                        font-weight: bold;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    "
                    onmouseover="this.style.background='linear-gradient(135deg, #1e7e34, #155724)'"
                    onmouseout="this.style.background='linear-gradient(135deg, #28a745, #1e7e34)'"
                >
                    <i class="fas fa-mouse-pointer"></i> Select All
                </button>
            </div>
            
            <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 15px;">
                <p style="margin: 0 0 10px 0; color: #ccc; font-size: 0.9em; text-align: center;">
                    Share your Trump score:
                </p>
                <div style="display: flex; gap: 10px; justify-content: center;">
                    <a href="https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent('🎯 Check out my Trump Tweet Challenge score! Can you beat it?')}" target="_blank" style="color: #1da1f2; text-decoration: none; padding: 8px 16px; background: rgba(29,161,242,0.1); border-radius: 8px; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(29,161,242,0.2)'" onmouseout="this.style.background='rgba(29,161,242,0.1)'">
                        <i class="fab fa-twitter"></i> Tweet
                    </a>
                    <a href="https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}" target="_blank" style="color: #4267B2; text-decoration: none; padding: 8px 16px; background: rgba(66,103,178,0.1); border-radius: 8px; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(66,103,178,0.2)'" onmouseout="this.style.background='rgba(66,103,178,0.1)'">
                        <i class="fab fa-facebook-f"></i> Share
                    </a>
                </div>
            </div>
            
            <button 
                onclick="this.parentElement.parentElement.remove()" 
                style="
                    position: absolute; 
                    top: 10px; 
                    right: 15px; 
                    background: none; 
                    border: none; 
                    color: #ccc; 
                    font-size: 20px; 
                    cursor: pointer; 
                    padding: 5px;
                    transition: color 0.3s ease;
                " 
                onmouseover="this.style.color='white'" 
                onmouseout="this.style.color='#ccc'"
            >×</button>
        </div>
    `;
    
    document.body.appendChild(shareUrlDisplay);
    
    // Auto-select the URL for easy copying
    setTimeout(() => {
        const input = document.getElementById('shareUrlInput');
        if (input) {
            input.focus();
            input.select();
        }
    }, 100);
    
    // Auto-remove after 30 seconds
    setTimeout(() => {
        if (shareUrlDisplay.parentElement) {
            shareUrlDisplay.remove();
        }
    }, 30000);
}

function copyShareUrlFromField() {
    const input = document.getElementById('shareUrlInput');
    if (input) {
        input.select();
        input.setSelectionRange(0, 99999); // For mobile devices
        
        // Try to copy
        try {
            const successful = document.execCommand('copy');
            if (successful) {
                // Show success feedback
                const button = event.target;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                button.style.background = 'linear-gradient(135deg, #28a745, #1e7e34)';
                
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.style.background = 'linear-gradient(135deg, #007bff, #0056b3)';
                }, 2000);
            } else {
                throw new Error('Copy command failed');
            }
        } catch (err) {
            // Fallback: just keep it selected
            console.log('Copy failed, URL remains selected for manual copy');
            alert('Please manually copy the selected URL (Ctrl+C or Cmd+C)');
        }
    }
}

function selectAllShareUrl() {
    const input = document.getElementById('shareUrlInput');
    if (input) {
        input.focus();
        input.select();
        input.setSelectionRange(0, 99999); // For mobile devices
        
        // Show feedback
        const button = event.target;
        const originalText = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i> Selected!';
        button.style.background = 'linear-gradient(135deg, #17a2b8, #117a8b)';
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.background = 'linear-gradient(135deg, #28a745, #1e7e34)';
        }, 1500);
    }
}

function showShareOptions(shareUrl, copySuccessful = true) {
    // Create a temporary notification for additional sharing options
    const notification = document.createElement('div');
    notification.className = 'share-notification';
    
    const copyStatus = copySuccessful 
        ? '<div style="color: #28a745; margin-bottom: 10px;"><i class="fas fa-check"></i> Link copied to clipboard!</div>'
        : '<div style="color: #ffc107; margin-bottom: 10px;"><i class="fas fa-exclamation-triangle"></i> Click to copy link:</div>';
    
    const copyButton = copySuccessful 
        ? '' 
        : `<button onclick="manualCopyLink('${shareUrl}', this)" style="background: #007bff; color: white; border: none; padding: 8px 12px; border-radius: 5px; margin-bottom: 10px; width: 100%; font-size: 0.9em; cursor: pointer;">
               <i class="fas fa-copy"></i> Copy Share Link
           </button>`;
    
    notification.innerHTML = `
        <div style="background: rgba(0,0,0,0.95); color: white; padding: 20px; border-radius: 15px; position: fixed; top: 20px; right: 20px; z-index: 9999; max-width: 320px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 2px solid rgba(255,215,0,0.3);">
            <div style="margin-bottom: 10px;"><strong>🎉 Share link created!</strong></div>
            ${copyStatus}
            ${copyButton}
            <div style="font-size: 0.9em; margin-bottom: 15px; color: #ccc;">Share your Trump score:</div>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <a href="https://twitter.com/intent/tweet?url=${encodeURIComponent(shareUrl)}&text=${encodeURIComponent('🎯 Check out my Trump Tweet Challenge score! Can you beat it?')}" target="_blank" style="color: #1da1f2; text-decoration: none; padding: 8px 12px; background: rgba(29,161,242,0.1); border-radius: 8px; flex: 1; text-align: center; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(29,161,242,0.2)'" onmouseout="this.style.background='rgba(29,161,242,0.1)'">
                    <i class="fab fa-twitter"></i> Tweet
                </a>
                <a href="https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(shareUrl)}" target="_blank" style="color: #4267B2; text-decoration: none; padding: 8px 12px; background: rgba(66,103,178,0.1); border-radius: 8px; flex: 1; text-align: center; transition: all 0.3s ease;" onmouseover="this.style.background='rgba(66,103,178,0.2)'" onmouseout="this.style.background='rgba(66,103,178,0.1)'">
                    <i class="fab fa-facebook-f"></i> Share
                </a>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" style="position: absolute; top: 8px; right: 10px; background: none; border: none; color: #ccc; font-size: 18px; cursor: pointer; padding: 5px;" onmouseover="this.style.color='white'" onmouseout="this.style.color='#ccc'">×</button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 12 seconds (longer if manual copy needed)
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 12000);
}

function manualCopyLink(url, buttonElement) {
    copyToClipboard(url)
        .then(() => {
            buttonElement.innerHTML = '<i class="fas fa-check"></i> Copied!';
            buttonElement.style.background = '#28a745';
            setTimeout(() => {
                if (buttonElement.parentElement) {
                    buttonElement.innerHTML = '<i class="fas fa-copy"></i> Copy Share Link';
                    buttonElement.style.background = '#007bff';
                }
            }, 2000);
        })
        .catch(err => {
            console.error('Manual copy failed:', err);
            buttonElement.innerHTML = '<i class="fas fa-times"></i> Copy Failed';
            buttonElement.style.background = '#dc3545';
            
            // Show fallback - select the URL for manual copying
            showUrlForManualCopy(url);
        });
}

function showUrlForManualCopy(url) {
    // Create a text input with the URL selected for manual copying
    const input = document.createElement('input');
    input.type = 'text';
    input.value = url;
    input.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 10000;
        padding: 10px;
        font-size: 16px;
        border: 2px solid #007bff;
        border-radius: 5px;
        width: 80%;
        max-width: 500px;
        background: white;
        color: black;
    `;
    
    document.body.appendChild(input);
    input.select();
    input.focus();
    
    // Remove after 5 seconds or when user clicks elsewhere
    const cleanup = () => {
        if (input.parentElement) {
            input.remove();
        }
    };
    
    setTimeout(cleanup, 5000);
    input.addEventListener('blur', cleanup);
}

function quickCopyLink() {
    console.log('quickCopyLink called, currentShareUrl:', currentShareUrl); // Debug log
    
    if (!currentShareUrl) {
        alert('No share link available! Please create a share link first.');
        return;
    }
    
    const copyBtn = document.querySelector('.copy-link-btn');
    const originalText = copyBtn.innerHTML;
    
    copyBtn.disabled = true;
    copyBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Copying...';
    
    copyToClipboard(currentShareUrl)
        .then(() => {
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.style.background = 'linear-gradient(135deg, #28a745, #20c997)';
            
            // Show a brief success message
            const successMsg = document.createElement('div');
            successMsg.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: rgba(40, 167, 69, 0.95);
                color: white;
                padding: 15px 25px;
                border-radius: 10px;
                z-index: 10000;
                font-weight: bold;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            `;
            successMsg.innerHTML = '<i class="fas fa-check"></i> Link copied to clipboard!';
            document.body.appendChild(successMsg);
            
            setTimeout(() => {
                if (successMsg.parentElement) {
                    successMsg.remove();
                }
            }, 2000);
            
            // Reset button after delay
            setTimeout(() => {
                copyBtn.disabled = false;
                copyBtn.innerHTML = originalText;
                copyBtn.style.background = 'linear-gradient(135deg, #17a2b8, #007bff)';
            }, 2000);
        })
        .catch(err => {
            console.error('Quick copy failed:', err);
            copyBtn.innerHTML = '<i class="fas fa-times"></i> Failed';
            copyBtn.style.background = 'linear-gradient(135deg, #dc3545, #c82333)';
            
            // Reset button after delay
            setTimeout(() => {
                copyBtn.disabled = false;
                copyBtn.innerHTML = originalText;
                copyBtn.style.background = 'linear-gradient(135deg, #17a2b8, #007bff)';
            }, 2000);
            
            // Show fallback
            showUrlForManualCopy(currentShareUrl);
        });
}

// Enhanced features loaded
console.log('🎯 Post Like Trump Challenge loaded successfully!');
