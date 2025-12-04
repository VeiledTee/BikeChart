import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_preprocess_data(filepath):
    """Load and clean the fitness data"""
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort by date
    df = df.sort_values('Date')

    # Convert time strings to minutes
    def time_to_minutes(time_str):
        if pd.isna(time_str) or time_str == '--':
            return np.nan
        try:
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 3:  # HH:MM:SS
                    return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
                elif len(parts) == 2:  # MM:SS
                    return int(parts[0]) + int(parts[1]) / 60
        except:
            return np.nan
        return np.nan

    # Apply conversion to time columns
    df['Time_minutes'] = df['Time'].apply(time_to_minutes)

    # Convert numeric columns, handling '--' as NaN
    numeric_columns = ['Distance', 'Calories', 'Avg HR', 'Max HR', 'Aerobic TE',
                       'Avg Speed', 'Max Speed', 'Total Ascent', 'Avg Bike Cadence',
                       'Max Bike Cadence', 'Normalized Power® (NP®)', 'Avg Power',
                       'Max Power', 'Training Stress Score®']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate efficiency metrics - intervals.icu style
    # Use Normalized Power when available, otherwise Avg Power
    if 'Normalized Power® (NP®)' in df.columns:
        df['Efficiency'] = df['Normalized Power® (NP®)'] / df['Avg HR']
    elif 'Avg Power' in df.columns:
        df['Efficiency'] = df['Avg Power'] / df['Avg HR']
    else:
        df['Efficiency'] = np.nan

    # Additional efficiency metrics
    df['Calories_per_hour'] = df['Calories'] / (df['Time_minutes'] / 60)
    df['Distance_per_hour'] = df['Distance'] / (df['Time_minutes'] / 60)

    # Add week number and month for aggregation
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.day_name()
    df['Day'] = df['Date'].dt.day

    return df


def plot_intervals_icu_efficiency(df):
    """Recreate intervals.icu style efficiency plot (Normalized Power / Avg HR over time)"""

    # Clean data - need both Avg HR and Normalized Power
    required_cols = ['Date', 'Avg HR']
    if 'Normalized Power® (NP®)' in df.columns:
        power_col = 'Normalized Power® (NP®)'
    elif 'Avg Power' in df.columns:
        power_col = 'Avg Power'
    else:
        print("⚠️ No power data available for efficiency plot")
        return plt.figure(figsize=(14, 8))

    df_clean = df.dropna(subset=['Date', 'Avg HR', power_col]).copy()

    if len(df_clean) == 0:
        print("⚠️ No data with both HR and power available")
        return plt.figure(figsize=(14, 8))

    # Calculate efficiency
    df_clean['Efficiency'] = df_clean[power_col] / df_clean['Avg HR']

    # Sort by date
    df_clean = df_clean.sort_values('Date')

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12),
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- MAIN EFFICIENCY PLOT (top) ---
    # Plot efficiency points
    scatter = ax1.scatter(df_clean['Date'], df_clean['Efficiency'],
                          s=100, alpha=0.8, color='#2E86AB', edgecolors='white', linewidth=1.5)

    # Add connecting lines between points
    ax1.plot(df_clean['Date'], df_clean['Efficiency'],
             color='#2E86AB', alpha=0.5, linewidth=1, linestyle='--')

    # Calculate and plot rolling average (7-day window)
    if len(df_clean) > 1:
        # Create a daily series for proper rolling average
        df_clean['Date_only'] = df_clean['Date'].dt.date
        daily_eff = df_clean.groupby('Date_only')['Efficiency'].mean().reset_index()
        daily_eff['Date_only'] = pd.to_datetime(daily_eff['Date_only'])
        daily_eff = daily_eff.sort_values('Date_only')

        # Calculate rolling average
        window = min(7, len(daily_eff))
        daily_eff['Rolling_Avg'] = daily_eff['Efficiency'].rolling(window=window, center=True, min_periods=1).mean()

        ax1.plot(daily_eff['Date_only'], daily_eff['Rolling_Avg'],
                 color='#D74E09', linewidth=3, label=f'{window}-day Rolling Avg')

    # Add horizontal line for average efficiency
    avg_efficiency = df_clean['Efficiency'].mean()
    ax1.axhline(y=avg_efficiency, color='#6C757D', linestyle='--', alpha=0.7,
                linewidth=2, label=f'Average: {avg_efficiency:.3f}')

    # Add trend line (linear regression)
    if len(df_clean) > 2:
        # Convert dates to numeric for regression
        dates_numeric = (df_clean['Date'] - df_clean['Date'].min()).dt.days
        mask = ~np.isnan(df_clean['Efficiency'].values)

        if mask.sum() > 1:
            try:
                x = dates_numeric[mask]
                y = df_clean['Efficiency'].values[mask]

                # Fit polynomial
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)

                # Create date range for trend line
                date_range = pd.date_range(start=df_clean['Date'].min(),
                                           end=df_clean['Date'].max(), periods=100)
                trend_dates_numeric = (date_range - df_clean['Date'].min()).days

                # Plot trend line
                ax1.plot(date_range, p(trend_dates_numeric),
                         color='#2D3047', linewidth=2.5, linestyle='-',
                         label=f'Trend: {"+" if z[0] > 0 else ""}{z[0] * 7:.3f}/week')
            except Exception as e:
                print(f"⚠️ Could not calculate trend: {e}")

    # Customize main plot
    ax1.set_title('Cycling Efficiency Over Time (Normalized Power / Avg HR)',
                  fontsize=18, fontweight='bold', pad=20, color='#1f2937')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Efficiency (Watts per BPM)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(loc='upper left', fontsize=10)

    # Add value annotations for each point
    for i, (date, eff) in enumerate(zip(df_clean['Date'], df_clean['Efficiency'])):
        if not pd.isna(eff):
            # Only label every other point if there are many
            if len(df_clean) <= 15 or i % 2 == 0:
                ax1.annotate(f'{eff:.3f}',
                             xy=(date, eff),
                             xytext=(0, 8),
                             textcoords='offset points',
                             ha='center',
                             fontsize=8,
                             alpha=0.7,
                             color='#2E86AB')

    # Set y-axis limits with some padding
    y_min = df_clean['Efficiency'].min() * 0.95
    y_max = df_clean['Efficiency'].max() * 1.05
    ax1.set_ylim(y_min, y_max)

    # --- HISTOGRAM OF EFFICIENCY DISTRIBUTION (bottom) ---
    # Create histogram
    n, bins, patches = ax2.hist(df_clean['Efficiency'].dropna(), bins=15,
                                color='#2E86AB', alpha=0.7, edgecolor='black')

    # Add vertical line for average
    ax2.axvline(x=avg_efficiency, color='#D74E09', linestyle='--',
                linewidth=2, label=f'Avg: {avg_efficiency:.3f}')

    # Add vertical line for median
    median_efficiency = df_clean['Efficiency'].median()
    ax2.axvline(x=median_efficiency, color='#2D3047', linestyle=':',
                linewidth=2, label=f'Median: {median_efficiency:.3f}')

    # Customize histogram
    ax2.set_title('Efficiency Distribution', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('Efficiency (Watts per BPM)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle=':')

    # Add value labels on histogram bars
    for i in range(len(patches)):
        height = patches[i].get_height()
        if height > 0:
            ax2.text(patches[i].get_x() + patches[i].get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_daily_efficiency_timeline(df):
    """Create a detailed daily efficiency timeline similar to intervals.icu"""

    # Clean data
    if 'Normalized Power® (NP®)' in df.columns:
        power_col = 'Normalized Power® (NP®)'
    elif 'Avg Power' in df.columns:
        power_col = 'Avg Power'
    else:
        print("⚠️ No power data available")
        return plt.figure(figsize=(14, 8))

    df_clean = df.dropna(subset=['Date', 'Avg HR', power_col]).copy()

    if len(df_clean) == 0:
        print("⚠️ No data with both HR and power available")
        return plt.figure(figsize=(14, 8))

    # Calculate efficiency
    df_clean['Efficiency'] = df_clean[power_col] / df_clean['Avg HR']

    # Sort by date
    df_clean = df_clean.sort_values('Date')

    # Create daily aggregates
    df_clean['Date_only'] = df_clean['Date'].dt.date
    daily_stats = df_clean.groupby('Date_only').agg({
        'Efficiency': 'mean',
        'Avg HR': 'mean',
        power_col: 'mean',
        'Title': 'count'  # Number of activities that day
    }).reset_index()

    daily_stats['Date_only'] = pd.to_datetime(daily_stats['Date_only'])
    daily_stats = daily_stats.sort_values('Date_only')

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14),
                                        gridspec_kw={'height_ratios': [2, 1, 1]})

    # --- Plot 1: Efficiency Timeline ---
    # Create a continuous date range
    start_date = daily_stats['Date_only'].min()
    end_date = daily_stats['Date_only'].max()
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Create a DataFrame with all dates
    timeline_df = pd.DataFrame({'Date': all_dates})
    timeline_df = timeline_df.merge(daily_stats, left_on='Date', right_on='Date_only', how='left')

    # Plot efficiency as bars
    colors = ['#4CAF50' if eff >= daily_stats['Efficiency'].mean() else '#F44336'
              for eff in timeline_df['Efficiency']]

    bars = ax1.bar(timeline_df['Date'], timeline_df['Efficiency'],
                   color=colors, alpha=0.7, edgecolor='white', linewidth=0.5)

    # Add average line
    avg_eff = daily_stats['Efficiency'].mean()
    ax1.axhline(y=avg_eff, color='#333333', linestyle='--', linewidth=2,
                label=f'Average: {avg_eff:.3f}')

    # Add zero-width bars for missing days (gaps)
    missing_days = timeline_df[timeline_df['Efficiency'].isna()]
    if len(missing_days) > 0:
        ax1.bar(missing_days['Date'], [avg_eff * 0.01] * len(missing_days),  # Tiny bars
                color='#E0E0E0', alpha=0.3, edgecolor='none')

    # Customize
    ax1.set_title('Daily Cycling Efficiency Timeline', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Efficiency (W/BPM)', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.legend(loc='upper left')

    # Add value labels on bars
    for bar, date, eff in zip(bars, timeline_df['Date'], timeline_df['Efficiency']):
        if not pd.isna(eff) and bar.get_height() > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{eff:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)

    # --- Plot 2: Power and HR Components ---
    # Plot power
    color1 = 'tab:blue'
    ax2.plot(daily_stats['Date_only'], daily_stats[power_col],
             'o-', color=color1, linewidth=2, markersize=6, label='Power (W)')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Power (Watts)', color=color1, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # Plot HR on secondary axis
    ax2b = ax2.twinx()
    color2 = 'tab:red'
    ax2b.plot(daily_stats['Date_only'], daily_stats['Avg HR'],
              's--', color=color2, linewidth=2, markersize=6, label='Avg HR (BPM)')
    ax2b.set_ylabel('Avg Heart Rate (BPM)', color=color2, fontsize=12)
    ax2b.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax2.set_title('Power and Heart Rate Components', fontsize=14, fontweight='bold', pad=10)

    # --- Plot 3: Efficiency Statistics ---
    # Calculate weekly averages
    daily_stats['Week'] = daily_stats['Date_only'].dt.isocalendar().week
    weekly_stats = daily_stats.groupby('Week').agg({
        'Efficiency': 'mean',
        'Title': 'sum'  # Count of activities
    }).reset_index()

    # Bar width
    width = 0.35
    x = np.arange(len(weekly_stats))

    # Plot efficiency bars
    bars1 = ax3.bar(x - width / 2, weekly_stats['Efficiency'], width,
                    label='Weekly Avg Efficiency', color='#4CAF50', alpha=0.7)

    # Plot activity count on secondary axis
    ax3b = ax3.twinx()
    bars2 = ax3b.bar(x + width / 2, weekly_stats['Title'], width,
                     label='Activities per Week', color='#2196F3', alpha=0.7)

    # Customize
    ax3.set_xlabel('Week Number', fontsize=12)
    ax3.set_ylabel('Avg Efficiency (W/BPM)', color='#4CAF50', fontsize=12)
    ax3.tick_params(axis='y', labelcolor='#4CAF50')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Week {w}' for w in weekly_stats['Week']], rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')

    ax3b.set_ylabel('Number of Activities', color='#2196F3', fontsize=12)
    ax3b.tick_params(axis='y', labelcolor='#2196F3')

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax3.set_title('Weekly Efficiency & Activity Volume', fontsize=14, fontweight='bold', pad=10)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        if not pd.isna(height):
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        if not pd.isna(height):
            ax3b.text(bar.get_x() + bar.get_width() / 2., height,
                      f'{int(height)}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('CYCLING EFFICIENCY ANALYSIS - INTERVALS.ICU STYLE',
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_power_hr_efficiency_by_week(df):
    """Plot Power vs HR colored by week with trendlines for each week"""

    # --- DATA CLEANING & PREPARATION ---
    REQUIRED_COLS = ['Date', 'Avg HR', 'Avg Power']

    # Check if we have the required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        print(f"⚠️ Missing columns for efficiency plot: {missing_cols}")
        # Try with Normalized Power if Avg Power is missing
        if 'Avg Power' not in df.columns and 'Normalized Power® (NP®)' in df.columns:
            print("Using Normalized Power instead of Avg Power")
            df_plot = df.copy()
            df_plot['Avg Power'] = df_plot['Normalized Power® (NP®)']
        else:
            return plt.figure(figsize=(12, 8))
    else:
        df_plot = df.copy()

    # Drop rows with missing values in required columns
    df_plot = df_plot.dropna(subset=['Date', 'Avg HR', 'Avg Power'])

    if len(df_plot) < 2:
        print("⚠️ Not enough data for efficiency plot")
        return plt.figure(figsize=(12, 8))

    # Convert to appropriate types
    df_plot.loc[:, 'Date'] = pd.to_datetime(df_plot['Date'])
    for col in ['Avg HR', 'Avg Power']:
        df_plot.loc[:, col] = pd.to_numeric(df_plot[col], errors='coerce')

    # Drop any rows that failed conversion
    df_plot = df_plot.dropna(subset=['Avg HR', 'Avg Power'])
    df_plot = df_plot.sort_values(by='Date').reset_index(drop=True)

    # Calculate the Time Index (Weeks) relative to the first activity date
    first_date = df_plot['Date'].min()
    df_plot.loc[:, 'Days_Since_Start'] = (df_plot['Date'] - first_date).dt.days
    # Calculate the Week Number (starting from 0)
    df_plot.loc[:, 'Week_Number'] = (df_plot['Days_Since_Start'] / 7).astype(int)

    # --- COMPLEX EFFICIENCY CHART GENERATION (Power vs. HR Colored by Time) ---

    # Set up the plot aesthetics
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define Colormap: Using 'plasma' which goes from red (early) to purple (late)
    cmap = get_cmap('plasma')
    norm = Normalize(vmin=df_plot['Week_Number'].min(), vmax=df_plot['Week_Number'].max())

    # Scatter Plot: Color points by Week Number
    scatter = ax.scatter(
        df_plot['Avg HR'],
        df_plot['Avg Power'],
        c=df_plot['Week_Number'],
        cmap=cmap,
        norm=norm,
        s=150,
        marker='s',  # Use squares as shown in the sample image
        alpha=0.9,
        edgecolors='white',
        linewidth=1,
        zorder=5,
        label='Activity Data Point'
    )

    # Plot Trendlines for each Week (only if at least 2 points in the week)
    week_trendlines_drawn = 0
    for week in sorted(df_plot['Week_Number'].unique()):
        # Filter data for the current week
        weekly_data = df_plot[df_plot['Week_Number'] == week]

        if len(weekly_data) >= 2:  # Need at least 2 points to draw a line
            x_week = weekly_data['Avg HR'].values
            y_week = weekly_data['Avg Power'].values

            # Only proceed if we have valid data
            mask = ~np.isnan(x_week) & ~np.isnan(y_week)
            x_week_clean = x_week[mask]
            y_week_clean = y_week[mask]

            if len(x_week_clean) >= 2 and len(y_week_clean) >= 2:
                try:
                    # Calculate best-fit line (Degree 1)
                    m, b = np.polyfit(x_week_clean, y_week_clean, 1)

                    # Create a full range of HR values for the line extension
                    x_range = np.array([
                        df_plot['Avg HR'].min() - 5,
                        df_plot['Avg HR'].max() + 5
                    ])
                    trendline = m * x_range + b

                    # Get the color for this specific week from the colormap
                    line_color = cmap(norm(week))

                    # Plot the trendline
                    ax.plot(
                        x_range,
                        trendline,
                        color=line_color,
                        linestyle=':',
                        linewidth=2,
                        zorder=2,
                        alpha=0.7
                    )
                    week_trendlines_drawn += 1
                except Exception as e:
                    print(f"⚠️ Could not fit trendline for week {week}: {e}")

    print(f"✓ Drew trendlines for {week_trendlines_drawn} weeks")

    # --- CUSTOMIZATION & LABELS ---

    # Title and axis labels
    ax.set_title('Power-to-HR Efficiency Over Time (Weeks)',
                 fontsize=18, fontweight='bold', color='#1f2937', pad=20)
    ax.set_xlabel('Avg. Heart Rate (bpm) - Effort', fontsize=14)
    ax.set_ylabel('Avg. Power (Watts) - Output', fontsize=14)

    # Add grid and adjust ticks
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=1)

    # Color Bar (Time in Weeks)
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Time (Weeks since first activity)',
                   rotation=270, labelpad=25, fontsize=14)

    # Set colorbar ticks to show each week
    week_ticks = np.arange(df_plot['Week_Number'].min(), df_plot['Week_Number'].max() + 1)
    if len(week_ticks) <= 10:  # Only show all ticks if not too many
        cbar.set_ticks(week_ticks)
        cbar.set_ticklabels([f'Week {int(w)}' for w in week_ticks])
    else:
        # Show only selected weeks
        selected_weeks = week_ticks[::max(1, len(week_ticks) // 5)]
        cbar.set_ticks(selected_weeks)
        cbar.set_ticklabels([f'Week {int(w)}' for w in selected_weeks])

    # Add subtle background color
    fig.patch.set_facecolor('#f3f4f6')
    ax.set_facecolor('white')

    # Add annotation for efficiency interpretation
    ax.text(0.02, 0.98,
            '↑ Power at same HR = Improved Efficiency\n→ Higher HR for same Power = More Effort',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Calculate and display overall efficiency trend
    if len(df_plot) > 1:
        try:
            # Fit overall trend
            x_all = df_plot['Avg HR'].values
            y_all = df_plot['Avg Power'].values
            mask = ~np.isnan(x_all) & ~np.isnan(y_all)
            if mask.sum() > 1:
                m_all, b_all = np.polyfit(x_all[mask], y_all[mask], 1)
                efficiency_slope = m_all

                # Add overall trend line
                x_range_all = np.linspace(df_plot['Avg HR'].min(), df_plot['Avg HR'].max(), 100)
                ax.plot(x_range_all, m_all * x_range_all + b_all,
                        color='black', linestyle='--', linewidth=3,
                        label=f'Overall Trend: {efficiency_slope:.2f} W/bpm',
                        zorder=3, alpha=0.8)

                # Add legend for overall trend
                ax.legend(loc='lower right', fontsize=10)
        except Exception as e:
            print(f"⚠️ Could not calculate overall trend: {e}")

    # Adjust layout
    plt.tight_layout()

    return fig


def plot_simple_power_hr_trends(df):
    """Simple and robust plot of Activity Date vs Normalized Power / Avg HR"""

    # Create a clean dataframe without NaN in our key columns
    df_clean = df.dropna(subset=['Date', 'Avg HR', 'Normalized Power® (NP®)']).copy()

    if len(df_clean) == 0:
        print("⚠️ No data with both Avg HR and Normalized Power available")
        return plt.figure(figsize=(12, 8))

    # Calculate Power/HR ratio
    df_clean['Power_to_HR_ratio'] = df_clean['Normalized Power® (NP®)'] / df_clean['Avg HR']

    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Normalized Power over time
    ax1.plot(df_clean['Date'], df_clean['Normalized Power® (NP®)'], 'bo-',
             linewidth=2, markersize=8, label='Normalized Power')
    ax1.set_xlabel('Activity Date')
    ax1.set_ylabel('Normalized Power (Watts)', color='blue')
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_title('Normalized Power Over Time', fontweight='bold')

    # Add value labels
    for date, value in zip(df_clean['Date'], df_clean['Normalized Power® (NP®)']):
        ax1.annotate(f'{int(value)}W', xy=(date, value), xytext=(0, 5),
                     textcoords='offset points', ha='center', fontsize=8)

    # Plot 2: Average HR over time
    ax2.plot(df_clean['Date'], df_clean['Avg HR'], 'ro-',
             linewidth=2, markersize=8, label='Average HR')
    ax2.set_xlabel('Activity Date')
    ax2.set_ylabel('Average HR (BPM)', color='red')
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    ax2.set_title('Average Heart Rate Over Time', fontweight='bold')

    # Add value labels
    for date, value in zip(df_clean['Date'], df_clean['Avg HR']):
        ax2.annotate(f'{int(value)}', xy=(date, value), xytext=(0, 5),
                     textcoords='offset points', ha='center', fontsize=8)

    # Plot 3: Power/HR Ratio (Efficiency) over time
    ax3.plot(df_clean['Date'], df_clean['Power_to_HR_ratio'], 'go-',
             linewidth=2, markersize=8, label='Power/HR Ratio')
    ax3.set_xlabel('Activity Date')
    ax3.set_ylabel('Watts per BPM', color='green')
    ax3.tick_params(axis='x', rotation=45)
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')
    ax3.set_title('Power to Heart Rate Ratio (Higher = More Efficient)', fontweight='bold')

    # Add value labels
    for date, value in zip(df_clean['Date'], df_clean['Power_to_HR_ratio']):
        ax3.annotate(f'{value:.2f}', xy=(date, value), xytext=(0, 5),
                     textcoords='offset points', ha='center', fontsize=8)

    plt.suptitle('POWER & HEART RATE ANALYSIS - ACTIVITY DATE VS METRICS',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def generate_efficiency_report(df):
    """Generate a detailed report on efficiency metrics"""
    print("=" * 70)
    print("CYCLING EFFICIENCY REPORT (intervals.icu style)")
    print("=" * 70)

    # Check available columns
    has_normalized_power = 'Normalized Power® (NP®)' in df.columns
    has_avg_hr = 'Avg HR' in df.columns

    if not (has_normalized_power and has_avg_hr):
        print("⚠️ Missing required columns for efficiency calculation")
        print(f"   Normalized Power available: {has_normalized_power}")
        print(f"   Avg HR available: {has_avg_hr}")
        return

    # Calculate efficiency
    df_clean = df.dropna(subset=['Normalized Power® (NP®)', 'Avg HR']).copy()
    df_clean['Efficiency'] = df_clean['Normalized Power® (NP®)'] / df_clean['Avg HR']

    if len(df_clean) == 0:
        print("⚠️ No data with both Normalized Power and Avg HR")
        return

    # Basic stats
    print(f"\n📊 BASIC STATISTICS:")
    print(f"   Total workouts with efficiency data: {len(df_clean)}")
    print(f"   Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")

    # Efficiency statistics
    print(f"\n📈 EFFICIENCY METRICS:")
    print(f"   Average Efficiency: {df_clean['Efficiency'].mean():.4f} W/BPM")
    print(f"   Median Efficiency: {df_clean['Efficiency'].median():.4f} W/BPM")
    print(f"   Best Efficiency: {df_clean['Efficiency'].max():.4f} W/BPM")
    print(f"   Worst Efficiency: {df_clean['Efficiency'].min():.4f} W/BPM")
    print(f"   Standard Deviation: {df_clean['Efficiency'].std():.4f} W/BPM")

    # Trend analysis
    if len(df_clean) >= 2:
        df_sorted = df_clean.sort_values('Date')
        first_half = df_sorted.iloc[:len(df_sorted) // 2]
        second_half = df_sorted.iloc[len(df_sorted) // 2:]

        early_avg = first_half['Efficiency'].mean()
        late_avg = second_half['Efficiency'].mean()
        change = late_avg - early_avg
        pct_change = (change / early_avg * 100) if early_avg != 0 else 0

        print(f"\n📊 TREND ANALYSIS:")
        print(f"   Early Period Avg: {early_avg:.4f} W/BPM")
        print(f"   Late Period Avg: {late_avg:.4f} W/BPM")
        print(f"   Change: {change:+.4f} W/BPM ({pct_change:+.1f}%)")

        if pct_change > 5:
            print("   ✅ Significant improvement in efficiency!")
        elif pct_change < -5:
            print("   ⚠️  Efficiency has declined - consider recovery or technique focus")
        else:
            print("   ↔️  Efficiency stable")

    # Efficiency zones
    eff_values = df_clean['Efficiency'].values
    print(f"\n🎯 EFFICIENCY DISTRIBUTION:")

    # Define efficiency zones based on typical values
    zones = {
        'Excellent (>1.1)': np.sum(eff_values > 1.1),
        'Good (0.95-1.1)': np.sum((eff_values >= 0.95) & (eff_values <= 1.1)),
        'Average (0.85-0.95)': np.sum((eff_values >= 0.85) & (eff_values < 0.95)),
        'Needs Improvement (<0.85)': np.sum(eff_values < 0.85)
    }

    for zone_name, count in zones.items():
        percentage = (count / len(eff_values)) * 100
        print(f"   {zone_name}: {count} workouts ({percentage:.1f}%)")

    print(f"\n💡 RECOMMENDATIONS:")

    avg_efficiency = df_clean['Efficiency'].mean()
    if avg_efficiency < 0.9:
        print("   1. Focus on improving pedaling technique and cadence")
        print("   2. Consider strength training for better power output")
        print("   3. Work on maintaining consistent effort during rides")
    elif avg_efficiency > 1.05:
        print("   1. Excellent efficiency! Consider increasing training volume")
        print("   2. You might benefit from higher intensity intervals")
        print("   3. Focus on maintaining consistency")
    else:
        print("   1. Good efficiency level. Continue with current training")
        print("   2. Consider adding variety to your workouts")
        print("   3. Monitor recovery to prevent overtraining")

    print("=" * 70)


def main():
    csv_file_path = "Activities.csv"

    # Load and preprocess data
    print("Loading data...")
    df = load_and_preprocess_data(csv_file_path)

    print(f"Data loaded successfully!")
    print(f"Number of records: {len(df)}")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Generate efficiency report
    generate_efficiency_report(df)

    # Create plots
    print("\nGenerating visualizations...")

    # Plot 1: intervals.icu style efficiency plot
    fig1 = plot_intervals_icu_efficiency(df)
    fig1.savefig('intervals_icu_efficiency.png', dpi=300, bbox_inches='tight')
    print("✓ intervals.icu style efficiency plot saved as 'intervals_icu_efficiency.png'")

    # Plot 2: Daily efficiency timeline
    fig2 = plot_daily_efficiency_timeline(df)
    fig2.savefig('daily_efficiency_timeline.png', dpi=300, bbox_inches='tight')
    print("✓ Daily efficiency timeline saved as 'daily_efficiency_timeline.png'")

    # Plot 3: Power-to-HR Efficiency by Week
    fig3 = plot_power_hr_efficiency_by_week(df)
    fig3.savefig('power_hr_efficiency_by_week.png', dpi=300, bbox_inches='tight')
    print("✓ Power-to-HR Efficiency by Week saved as 'power_hr_efficiency_by_week.png'")

    # Plot 4: Simple Power vs HR trends
    fig4 = plot_simple_power_hr_trends(df)
    fig4.savefig('simple_power_hr_trends.png', dpi=300, bbox_inches='tight')
    print("✓ Simple power/HR trends saved as 'simple_power_hr_trends.png'")

    # Show all plots
    plt.show()

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nKey files generated:")
    print("  - intervals_icu_efficiency.png    : Efficiency (NP/HR) over time (like intervals.icu)")
    print("  - daily_efficiency_timeline.png   : Detailed daily timeline with gaps")
    print("  - power_hr_efficiency_by_week.png : Power vs HR colored by week")
    print("  - simple_power_hr_trends.png      : Power & HR components over time")
    print("\nThe intervals.icu style plot shows:")
    print("  • Efficiency = Normalized Power ÷ Average Heart Rate")
    print("  • Higher values = more power output per heart beat")
    print("  • Trend line shows overall improvement/decline")
    print("  • Rolling average smooths out daily fluctuations")
    print("  • Histogram shows distribution of your efficiency values")


if __name__ == "__main__":
    main()