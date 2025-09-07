"""
Analytics and Metrics Generation for Trump Tweet Classifier

This module provides comprehensive analytics including:
- Submission trends and patterns
- Geographic analysis
- User engagement metrics  
- Trump-level distribution
- Performance visualizations
"""

import sqlite3
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
import geoip2.database
import geoip2.errors
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class TrumpAnalytics:
    """Comprehensive analytics for the Trump Tweet Classifier"""
    
    def __init__(self, db_path: str = "data/trump_classifier.db"):
        self.db_path = db_path
        self.geoip_db = None
        self._setup_geoip()
    
    def _setup_geoip(self):
        """Initialize GeoIP database for geographic analysis"""
        try:
            # Try to find GeoLite2 database (user would need to download this)
            geoip_paths = [
                "/usr/share/GeoIP/GeoLite2-Country.mmdb",
                "./GeoLite2-Country.mmdb",
                "data/GeoLite2-Country.mmdb"
            ]
            
            for path in geoip_paths:
                if Path(path).exists():
                    self.geoip_db = geoip2.database.Reader(path)
                    logger.info(f"GeoIP database loaded from {path}")
                    break
            
            if not self.geoip_db:
                logger.warning("GeoIP database not found. Geographic analysis will be limited.")
                
        except Exception as e:
            logger.warning(f"Could not load GeoIP database: {e}")
    
    def get_db_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the application"""
        with self.get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Total submissions
            cursor.execute("SELECT COUNT(*) FROM submissions")
            total_submissions = cursor.fetchone()[0]
            
            # Unique users (by IP)
            cursor.execute("SELECT COUNT(DISTINCT user_ip) FROM submissions")
            unique_users = cursor.fetchone()[0]
            
            # Average confidence
            cursor.execute("SELECT AVG(confidence) FROM submissions WHERE confidence IS NOT NULL")
            avg_confidence = cursor.fetchone()[0] or 0
            
            # Trump percentage
            cursor.execute("SELECT AVG(CASE WHEN classification = 1 THEN 1.0 ELSE 0.0 END) * 100 FROM submissions WHERE classification IS NOT NULL")
            trump_percentage = cursor.fetchone()[0] or 0
            
            # Today's submissions
            cursor.execute("SELECT COUNT(*) FROM submissions WHERE DATE(created_at) = DATE('now')")
            today_submissions = cursor.fetchone()[0]
            
            # Feedback stats
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(CASE WHEN agrees_with_rating THEN 1.0 ELSE 0.0 END) * 100 FROM feedback")
            agreement_rate = cursor.fetchone()[0] or 0
            
            return {
                "total_submissions": total_submissions,
                "unique_users": unique_users,
                "avg_confidence": round(avg_confidence, 2),
                "trump_percentage": round(trump_percentage, 1),
                "today_submissions": today_submissions,
                "total_feedback": total_feedback,
                "agreement_rate": round(agreement_rate, 1),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def get_hourly_submissions(self, days: int = 7) -> str:
        """Generate plot of submissions per hour"""
        with self.get_db_connection() as conn:
            # Get hourly submission data
            query = """
            SELECT 
                DATE(created_at) as date,
                CAST(strftime('%H', created_at) AS INTEGER) as hour,
                COUNT(*) as submissions
            FROM submissions 
            WHERE created_at >= datetime('now', '-{} days')
            GROUP BY DATE(created_at), CAST(strftime('%H', created_at) AS INTEGER)
            ORDER BY date, hour
            """.format(days)
            
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return self._create_empty_plot("No submission data available")
        
        # Create the plot
        plt.figure(figsize=(14, 8))
        
        # Create a pivot table for heatmap
        if len(df) > 1:
            pivot_df = df.pivot_table(values='submissions', index='hour', columns='date', fill_value=0)
            
            # Create heatmap
            plt.subplot(2, 1, 1)
            sns.heatmap(pivot_df.T, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Submissions'})
            plt.title(f'Submissions Heatmap - Last {days} Days', fontsize=16, fontweight='bold')
            plt.ylabel('Date')
            plt.xlabel('Hour of Day')
        
        # Line plot for trend
        plt.subplot(2, 1, 2)
        hourly_avg = df.groupby('hour')['submissions'].mean()
        plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=3, markersize=8)
        plt.title('Average Submissions by Hour', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Submissions')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24))
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def get_trump_level_distribution(self) -> str:
        """Generate plot showing distribution of Trump levels"""
        with self.get_db_connection() as conn:
            query = """
            SELECT trump_level as level, COUNT(*) as count
            FROM submissions 
            WHERE trump_level IS NOT NULL
            GROUP BY trump_level
            ORDER BY 
                CASE trump_level
                    WHEN 'CERTIFIED Trump' THEN 6
                    WHEN 'Donald Trump Jr.' THEN 5
                    WHEN 'Eric Trump' THEN 4
                    WHEN 'Trump Supporter' THEN 3
                    WHEN 'Tiffany Trump' THEN 2
                    ELSE 1
                END DESC
            """
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return self._create_empty_plot("No level data available")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Color scheme for different levels
        colors = ['#ff6b6b', '#ffa500', '#ffed4a', '#38c172', '#3490dc', '#6574cd']
        
        # Bar plot
        plt.subplot(2, 1, 1)
        bars = plt.bar(df['level'], df['count'], color=colors[:len(df)])
        plt.title('Distribution of Trump Levels', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Submissions')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        plt.subplot(2, 1, 2)
        plt.pie(df['count'], labels=df['level'], colors=colors[:len(df)], autopct='%1.1f%%', startangle=90)
        plt.title('Trump Level Percentage Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def get_confidence_analysis(self) -> str:
        """Generate confidence score analysis plots"""
        with self.get_db_connection() as conn:
            query = """
            SELECT 
                confidence,
                classification,
                trump_level as level,
                created_at
            FROM submissions 
            WHERE confidence IS NOT NULL
            ORDER BY created_at
            """
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return self._create_empty_plot("No confidence data available")
        
        plt.figure(figsize=(15, 10))
        
        # Confidence distribution histogram
        plt.subplot(2, 3, 1)
        plt.hist(df['confidence'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.title('Confidence Score Distribution', fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Confidence by classification
        plt.subplot(2, 3, 2)
        trump_conf = df[df['classification'] == 1]['confidence']
        not_trump_conf = df[df['classification'] == 0]['confidence']
        
        plt.hist([trump_conf, not_trump_conf], bins=15, label=['Trump-like', 'Not Trump-like'], 
                color=['red', 'blue'], alpha=0.7)
        plt.title('Confidence by Classification', fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot by level
        plt.subplot(2, 3, 3)
        level_order = ['CERTIFIED Trump', 'Donald Trump Jr.', 'Eric Trump', 
                      'Trump Supporter', 'Tiffany Trump', 'Definitely Not Trump']
        df_clean = df[df['level'].isin(level_order)]
        
        if not df_clean.empty:
            sns.boxplot(data=df_clean, y='level', x='confidence', order=level_order)
            plt.title('Confidence by Trump Level', fontweight='bold')
        
        # Confidence over time
        plt.subplot(2, 3, 4)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df_daily = df.set_index('created_at').resample('D')['confidence'].mean()
        
        plt.plot(df_daily.index, df_daily.values, marker='o', linewidth=2)
        plt.title('Average Confidence Over Time', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Average Confidence')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # High vs Low confidence submissions
        plt.subplot(2, 3, 5)
        high_conf = len(df[df['confidence'] > 0.8])
        med_conf = len(df[(df['confidence'] > 0.5) & (df['confidence'] <= 0.8)])
        low_conf = len(df[df['confidence'] <= 0.5])
        
        categories = ['Low\n(≤50%)', 'Medium\n(50-80%)', 'High\n(>80%)']
        values = [low_conf, med_conf, high_conf]
        colors = ['#ff6b6b', '#ffa500', '#38c172']
        
        plt.bar(categories, values, color=colors)
        plt.title('Confidence Categories', fontweight='bold')
        plt.ylabel('Number of Submissions')
        
        # Add value labels
        for i, v in enumerate(values):
            plt.text(i, v + max(values)*0.01, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Correlation matrix placeholder
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, f'Total Submissions:\n{len(df)}\n\nAvg Confidence:\n{df["confidence"].mean():.2f}\n\nStd Dev:\n{df["confidence"].std():.2f}', 
                ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.title('Summary Statistics', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def get_geographic_analysis(self) -> str:
        """Generate geographic analysis of users"""
        if not self.geoip_db:
            return self._create_empty_plot("GeoIP database not available for geographic analysis")
        
        with self.get_db_connection() as conn:
            query = """
            SELECT 
                user_ip,
                AVG(confidence) as avg_confidence,
                AVG(CASE WHEN classification = 1 THEN 1.0 ELSE 0.0 END) * 100 as trump_percentage,
                COUNT(*) as submission_count
            FROM submissions 
            WHERE user_ip IS NOT NULL
            GROUP BY user_ip
            HAVING COUNT(*) > 0
            """
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return self._create_empty_plot("No geographic data available")
        
        # Get country information for each IP
        countries = []
        for ip in df['user_ip']:
            try:
                # In a real app, you'd want to hash/anonymize IPs for privacy
                response = self.geoip_db.country(ip)
                countries.append(response.country.name or 'Unknown')
            except (geoip2.errors.AddressNotFoundError, Exception):
                countries.append('Unknown')
        
        df['country'] = countries
        
        # Aggregate by country
        country_stats = df.groupby('country').agg({
            'avg_confidence': 'mean',
            'trump_percentage': 'mean', 
            'submission_count': 'sum'
        }).reset_index()
        
        # Filter out Unknown and countries with < 2 submissions
        country_stats = country_stats[
            (country_stats['country'] != 'Unknown') & 
            (country_stats['submission_count'] >= 2)
        ].sort_values('submission_count', ascending=False)
        
        if country_stats.empty:
            return self._create_empty_plot("Insufficient geographic data for analysis")
        
        plt.figure(figsize=(15, 10))
        
        # Submissions by country
        plt.subplot(2, 2, 1)
        top_countries = country_stats.head(10)
        bars = plt.bar(range(len(top_countries)), top_countries['submission_count'])
        plt.title('Submissions by Country (Top 10)', fontweight='bold')
        plt.xlabel('Country')
        plt.ylabel('Number of Submissions')
        plt.xticks(range(len(top_countries)), top_countries['country'], rotation=45, ha='right')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Trump percentage by country
        plt.subplot(2, 2, 2)
        plt.scatter(top_countries['submission_count'], top_countries['trump_percentage'], 
                   s=100, alpha=0.7, c=top_countries['trump_percentage'], cmap='RdYlBu_r')
        plt.title('Trump Percentage vs Submissions by Country', fontweight='bold')
        plt.xlabel('Number of Submissions')
        plt.ylabel('Trump Percentage (%)')
        plt.colorbar(label='Trump %')
        
        # Confidence by country
        plt.subplot(2, 2, 3)
        plt.barh(range(len(top_countries)), top_countries['avg_confidence'])
        plt.title('Average Confidence by Country', fontweight='bold')
        plt.xlabel('Average Confidence')
        plt.ylabel('Country')
        plt.yticks(range(len(top_countries)), top_countries['country'])
        
        # Summary stats
        plt.subplot(2, 2, 4)
        total_countries = len(country_stats)
        avg_trump_global = country_stats['trump_percentage'].mean()
        avg_conf_global = country_stats['avg_confidence'].mean()
        
        plt.text(0.5, 0.5, f'Countries Analyzed: {total_countries}\n\n'
                          f'Global Avg Trump %: {avg_trump_global:.1f}%\n\n'
                          f'Global Avg Confidence: {avg_conf_global:.2f}\n\n'
                          f'Most Active: {top_countries.iloc[0]["country"]}\n'
                          f'({int(top_countries.iloc[0]["submission_count"])} submissions)', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.title('Geographic Summary', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def get_feedback_analysis(self) -> str:
        """Generate feedback and user engagement analysis"""
        with self.get_db_connection() as conn:
            # Get feedback data with submission details
            query = """
            SELECT 
                f.agrees_with_rating as agree,
                f.feedback_message as message,
                f.created_at as feedback_date,
                s.confidence,
                s.classification,
                s.trump_level as level,
                s.created_at as submission_date
            FROM feedback f
            JOIN submissions s ON f.submission_id = s.id
            ORDER BY f.created_at
            """
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return self._create_empty_plot("No feedback data available")
        
        plt.figure(figsize=(15, 10))
        
        # Agreement rate over time
        plt.subplot(2, 3, 1)
        df['feedback_date'] = pd.to_datetime(df['feedback_date'])
        daily_agreement = df.set_index('feedback_date').resample('D')['agree'].mean() * 100
        
        plt.plot(daily_agreement.index, daily_agreement.values, marker='o', linewidth=2, color='green')
        plt.title('Daily Agreement Rate', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Agreement Rate (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Agreement by confidence level
        plt.subplot(2, 3, 2)
        conf_bins = pd.cut(df['confidence'], bins=[0, 0.5, 0.7, 0.9, 1.0], labels=['Low', 'Medium', 'High', 'Very High'])
        agreement_by_conf = df.groupby(conf_bins)['agree'].mean() * 100
        
        bars = plt.bar(agreement_by_conf.index, agreement_by_conf.values, color=['red', 'orange', 'lightgreen', 'green'])
        plt.title('Agreement Rate by Confidence Level', fontweight='bold')
        plt.xlabel('Confidence Level')
        plt.ylabel('Agreement Rate (%)')
        plt.ylim(0, 100)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        # Agreement by Trump level
        plt.subplot(2, 3, 3)
        level_agreement = df.groupby('level')['agree'].mean() * 100
        level_agreement = level_agreement.sort_values(ascending=False)
        
        plt.barh(range(len(level_agreement)), level_agreement.values)
        plt.title('Agreement Rate by Trump Level', fontweight='bold')
        plt.xlabel('Agreement Rate (%)')
        plt.ylabel('Trump Level')
        plt.yticks(range(len(level_agreement)), level_agreement.index)
        plt.xlim(0, 100)
        
        # Feedback volume over time
        plt.subplot(2, 3, 4)
        daily_feedback = df.set_index('feedback_date').resample('D').size()
        
        plt.bar(daily_feedback.index, daily_feedback.values, alpha=0.7, color='purple')
        plt.title('Daily Feedback Volume', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Number of Feedback')
        plt.xticks(rotation=45)
        
        # Confidence distribution for agree vs disagree
        plt.subplot(2, 3, 5)
        agree_conf = df[df['agree']]['confidence']
        disagree_conf = df[~df['agree']]['confidence']
        
        plt.hist([agree_conf, disagree_conf], bins=15, label=['Agree', 'Disagree'], 
                alpha=0.7, color=['green', 'red'])
        plt.title('Confidence Distribution by Agreement', fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 3, 6)
        total_feedback = len(df)
        overall_agreement = df['agree'].mean() * 100
        avg_conf_agree = agree_conf.mean()
        avg_conf_disagree = disagree_conf.mean()
        
        plt.text(0.5, 0.5, f'Total Feedback: {total_feedback}\n\n'
                          f'Overall Agreement: {overall_agreement:.1f}%\n\n'
                          f'Avg Confidence (Agree): {avg_conf_agree:.2f}\n\n'
                          f'Avg Confidence (Disagree): {avg_conf_disagree:.2f}\n\n'
                          f'Feedback Rate: {(total_feedback/self.get_basic_stats()["total_submissions"]*100):.1f}%', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.title('Feedback Summary', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def get_sharing_analysis(self) -> str:
        """Generate sharing and viral metrics analysis"""
        with self.get_db_connection() as conn:
            query = """
            SELECT 
                is_public,
                share_hash,
                share_title,
                trump_level as level,
                confidence,
                classification,
                created_at
            FROM submissions 
            WHERE share_hash IS NOT NULL
            ORDER BY created_at
            """
            df = pd.read_sql_query(query, conn)
        
        if df.empty:
            return self._create_empty_plot("No sharing data available")
        
        plt.figure(figsize=(12, 8))
        
        # Sharing rate over time
        plt.subplot(2, 2, 1)
        df['created_at'] = pd.to_datetime(df['created_at'])
        daily_shares = df.set_index('created_at').resample('D').size()
        
        plt.plot(daily_shares.index, daily_shares.values, marker='o', linewidth=2, color='blue')
        plt.title('Daily Sharing Activity', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Number of Shares')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Most shared Trump levels
        plt.subplot(2, 2, 2)
        level_shares = df['level'].value_counts()
        
        plt.pie(level_shares.values, labels=level_shares.index, autopct='%1.1f%%', startangle=90)
        plt.title('Shared Content by Trump Level', fontweight='bold')
        
        # Confidence of shared content
        plt.subplot(2, 2, 3)
        plt.hist(df['confidence'], bins=15, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Confidence Distribution of Shared Content', fontweight='bold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Summary stats
        plt.subplot(2, 2, 4)
        total_shares = len(df)
        avg_conf_shared = df['confidence'].mean()
        trump_share_rate = (df['classification'].sum() / len(df)) * 100
        
        plt.text(0.5, 0.5, f'Total Shares: {total_shares}\n\n'
                          f'Avg Confidence (Shared): {avg_conf_shared:.2f}\n\n'
                          f'Trump Content Shared: {trump_share_rate:.1f}%\n\n'
                          f'Most Shared Level:\n{level_shares.index[0] if not level_shares.empty else "N/A"}', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
        plt.title('Sharing Summary', fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        return self._plot_to_base64()
    
    def _plot_to_base64(self) -> str:
        """Convert current matplotlib plot to base64 string"""
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.read()).decode()
        plt.close()  # Clean up
        return img_str
    
    def _create_empty_plot(self, message: str) -> str:
        """Create an empty plot with a message"""
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, message, ha='center', va='center', fontsize=16,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.axis('off')
        plt.title('No Data Available', fontsize=18, fontweight='bold')
        return self._plot_to_base64()

    def __del__(self):
        """Clean up GeoIP database connection"""
        if self.geoip_db:
            try:
                self.geoip_db.close()
            except:
                pass

