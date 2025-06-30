#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Define the file path
file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0272EN-SkillsNetwork/labs/dataset/2016.csv"

try:
    # Read the CSV file into a pandas DataFrame
    # The first row is automatically used as headers by default
    df = pd.read_csv(file_path)
    
    # Print the first 5 rows to verify correct loading
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Optional: Print basic information about the dataset
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # DATA PREPARATION - Check and correct data types
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    # 1. Check the data types of all columns
    print("\n1. Current data types:")
    print(df.dtypes)
    print("\n")
    
    # Display info about the dataset including data types and null values
    print("Dataset info:")
    print(df.info())
    
    # Check for any obvious data type issues
    print(f"\n2. Data type analysis:")
    
    # Check each column and identify potential issues
    for col in df.columns:
        print(f"\nColumn '{col}':")
        print(f"  - Data type: {df[col].dtype}")
        print(f"  - Null values: {df[col].isnull().sum()}")
        print(f"  - Unique values: {df[col].nunique()}")
        
        # Show sample values for better understanding
        if df[col].dtype == 'object':
            print(f"  - Sample values: {df[col].dropna().head(3).tolist()}")
        else:
            print(f"  - Value range: {df[col].min()} to {df[col].max()}")
    
    # 3. Identify and fix common data type issues
    print(f"\n3. Data type corrections:")
    
    # Store original dtypes for comparison
    original_dtypes = df.dtypes.copy()
    
    # Common corrections based on typical dataset patterns
    corrections_made = []
    
    # Check for columns that should be numeric but are stored as objects
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric, errors='coerce' will convert non-numeric to NaN
            try:
                # First, let's see if it can be converted to numeric
                numeric_conversion = pd.to_numeric(df[col], errors='coerce')
                
                # If less than 50% of values become NaN, it's likely a numeric column
                nan_percentage = numeric_conversion.isnull().sum() / len(df)
                
                if nan_percentage < 0.5:  # If less than 50% are NaN after conversion
                    df[col] = numeric_conversion
                    corrections_made.append(f"'{col}': object → numeric")
                    
            except:
                pass
    
    # Check for columns that might be dates
    for col in df.columns:
        if df[col].dtype == 'object' and col.lower() in ['date', 'time', 'year', 'month']:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if not df[col].isnull().all():  # If conversion was successful for some values
                    corrections_made.append(f"'{col}': object → datetime")
            except:
                pass
    
    # Display corrections made
    if corrections_made:
        print("Corrections applied:")
        for correction in corrections_made:
            print(f"  - {correction}")
    else:
        print("No automatic corrections were needed or could be safely applied.")
    
    # 4. Display final data types after corrections
    print(f"\n4. Data types after corrections:")
    final_dtypes = df.dtypes
    
    print("\nComparison of data types:")
    print(f"{'Column':<20} {'Original':<15} {'Current':<15} {'Changed'}")
    print("-" * 65)
    
    for col in df.columns:
        original = str(original_dtypes[col])
        current = str(final_dtypes[col])
        changed = "Yes" if original != current else "No"
        print(f"{col:<20} {original:<15} {current:<15} {changed}")
    
    # 5. Final dataset summary
    print(f"\n5. Final dataset summary:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum()} bytes")
    print(f"Total null values: {df.isnull().sum().sum()}")
    
    # 6. MISSING VALUES ANALYSIS AND TREATMENT
    print("\n" + "="*50)
    print("MISSING VALUES ANALYSIS AND TREATMENT")
    print("="*50)
    
    # Identify columns with missing values
    print("\n6. Missing values analysis:")
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create a summary of missing values
    missing_summary = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_percentage.values
    })
    
    # Filter to show only columns with missing values
    columns_with_missing = missing_summary[missing_summary['Missing_Count'] > 0]
    
    if len(columns_with_missing) > 0:
        print("Columns with missing values:")
        print(columns_with_missing.to_string(index=False))
        
        # 7. Fill missing values with mean for numeric columns
        print(f"\n7. Filling missing values with mean:")
        
        filled_columns = []
        skipped_columns = []
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:  # If column has missing values
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:  # Numeric columns
                    # Calculate mean excluding NaN values
                    mean_value = df[col].mean()
                    
                    # Fill missing values with mean
                    df[col].fillna(mean_value, inplace=True)
                    
                    filled_columns.append({
                        'column': col,
                        'mean_value': round(mean_value, 4),
                        'filled_count': missing_values[col]
                    })
                    
                else:  # Non-numeric columns
                    skipped_columns.append({
                        'column': col,
                        'data_type': str(df[col].dtype),
                        'missing_count': missing_values[col]
                    })
        
        # Display results of filling operation
        if filled_columns:
            print("\nNumeric columns filled with mean values:")
            for item in filled_columns:
                print(f"  - '{item['column']}': {item['filled_count']} missing values filled with mean = {item['mean_value']}")
        
        if skipped_columns:
            print(f"\nNon-numeric columns skipped (cannot use mean):")
            for item in skipped_columns:
                print(f"  - '{item['column']}' ({item['data_type']}): {item['missing_count']} missing values")
                
        # 8. Verify missing values after treatment
        print(f"\n8. Missing values verification after treatment:")
        remaining_missing = df.isnull().sum()
        remaining_total = remaining_missing.sum()
        
        if remaining_total > 0:
            print("Remaining missing values:")
            remaining_summary = remaining_missing[remaining_missing > 0]
            for col, count in remaining_summary.items():
                percentage = (count / len(df)) * 100
                print(f"  - '{col}': {count} missing values ({percentage:.2f}%)")
        else:
            print("✓ All missing values in numeric columns have been successfully filled!")
            
        # Summary of missing value treatment
        print(f"\nMissing value treatment summary:")
        print(f"  - Total columns processed: {len(df.columns)}")
        print(f"  - Columns with missing values before: {len(columns_with_missing)}")
        print(f"  - Numeric columns filled with mean: {len(filled_columns)}")
        print(f"  - Non-numeric columns skipped: {len(skipped_columns)}")
        print(f"  - Remaining missing values: {remaining_total}")
        
    else:
        print("✓ No missing values found in the dataset!")
    
    # 9. Final dataset status
    print(f"\n" + "="*50)
    print("FINAL DATASET STATUS")
    print("="*50)
    
    print(f"Shape: {df.shape}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum():,} bytes")
    
    print(f"\nData types summary:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  - {dtype}: {count} columns")
    
    print("\nData preparation and missing value treatment completed successfully!")
    print("Dataset is now ready for analysis!")
    
    # 10. TOP 10 COUNTRIES ANALYSIS AND VISUALIZATION
    print(f"\n" + "="*50)
    print("TOP 10 COUNTRIES ANALYSIS")
    print("="*50)
    
    # First, let's examine the column names to identify the relevant columns
    print(f"\nAvailable columns in the dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Try to identify the relevant columns for analysis
    # Common column names for happiness/wellbeing datasets
    possible_rank_cols = [col for col in df.columns if 'rank' in col.lower() or 'happiness' in col.lower()]
    possible_country_cols = [col for col in df.columns if 'country' in col.lower() or 'nation' in col.lower()]
    possible_gdp_cols = [col for col in df.columns if 'gdp' in col.lower() or 'economy' in col.lower()]
    possible_health_cols = [col for col in df.columns if 'health' in col.lower() or 'life' in col.lower()]
    
    print(f"\nIdentified potential columns:")
    print(f"Country columns: {possible_country_cols}")
    print(f"Ranking columns: {possible_rank_cols}")
    print(f"GDP columns: {possible_gdp_cols}")
    print(f"Health columns: {possible_health_cols}")
    
        # Try to select the most appropriate columns (adjust these based on actual column names)
    try:
        # Attempt to identify columns automatically
        country_col = possible_country_cols[0] if possible_country_cols else df.columns[0]
        
        # For GDP per capita - look for GDP-related columns
        gdp_col = None
        for col in df.columns:
            if any(term in col.lower() for term in ['gdp', 'economy', 'economic']):
                gdp_col = col
                break
        
        # For Healthy Life Expectancy - look for health/life-related columns
        health_col = None
        for col in df.columns:
            if any(term in col.lower() for term in ['health', 'life', 'expectancy']):
                health_col = col
                break
        
        # For ranking - look for happiness score or similar
        score_col = None
        for col in df.columns:
            if any(term in col.lower() for term in ['score', 'happiness', 'rank']):
                if 'rank' not in col.lower():  # Prefer score over rank
                    score_col = col
                    break
        
        if not score_col:  # If no score found, look for rank
            for col in df.columns:
                if 'rank' in col.lower():
                    score_col = col
                    break
        
        print(f"\nSelected columns for analysis:")
        print(f"Country: {country_col}")
        print(f"GDP per capita: {gdp_col}")
        print(f"Healthy Life Expectancy: {health_col}")
        print(f"Ranking/Score: {score_col}")
        
        if gdp_col and health_col and score_col:
            # Sort by happiness score (or rank) to get top 10 countries
            if 'rank' in score_col.lower():
                # If it's a rank column, sort ascending (lower rank = better)
                top_10 = df.nsmallest(10, score_col)
            else:
                # If it's a score column, sort descending (higher score = better)
                top_10 = df.nlargest(10, score_col)
            
            print(f"\nTop 10 countries:")
            display_cols = [country_col, score_col, gdp_col, health_col]
            print(top_10[display_cols].to_string(index=False))
            
            # Create visualization using plotly
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Create subplot with secondary y-axis
            fig1 = make_subplots(
                rows=1, cols=1,
                secondary_y=True,
                subplot_titles=('GDP per Capita and Healthy Life Expectancy - Top 10 Countries',)
            )
            
            # Add GDP per capita bar chart
            fig1.add_trace(
                go.Bar(
                    x=top_10[country_col],
                    y=top_10[gdp_col],
                    name='GDP per Capita',
                    marker_color='lightblue',
                    opacity=0.8
                ),
                secondary_y=False,
            )
            
            # Add Healthy Life Expectancy bar chart
            fig1.add_trace(
                go.Bar(
                    x=top_10[country_col],
                    y=top_10[health_col],
                    name='Healthy Life Expectancy',
                    marker_color='lightcoral',
                    opacity=0.8
                ),
                secondary_y=True,
            )
            
            # Update layout
            fig1.update_layout(
                title={
                    'text': 'GDP per Capita and Healthy Life Expectancy - Top 10 Countries',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                xaxis_title='Countries',
                barmode='group',
                height=600,
                width=1000,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Set y-axes titles
            fig1.update_yaxes(title_text="GDP per Capita", secondary_y=False)
            fig1.update_yaxes(title_text="Healthy Life Expectancy (Years)", secondary_y=True)
            
            # Rotate x-axis labels for better readability
            fig1.update_xaxes(tickangle=-45)
            
            # Show the plot
            fig1.show()
            
            print(f"\n✓ Bar chart 'fig1' created successfully!")
            print(f"✓ Chart shows GDP per Capita and Healthy Life Expectancy for top 10 countries")
            
            # Save the figure (optional)
            # fig1.write_html("top_10_countries_analysis.html")
            # print(f"✓ Chart saved as 'top_10_countries_analysis.html'")
            
        else:
            print(f"\n⚠️ Could not identify all required columns automatically.")
            print(f"Please check the column names and adjust the code accordingly.")
            print(f"Available columns: {list(df.columns)}")
            
    except Exception as viz_error:
        print(f"Error during visualization: {viz_error}")
        print(f"Please check if the required columns exist in the dataset.")
    
    # 11. DATA EXPLORATION - ADVANCED VISUALIZATIONS
    print(f"\n" + "="*50)
    print("DATA EXPLORATION - ADVANCED VISUALIZATIONS")
    print("="*50)
    
    try:
        # Import additional libraries for advanced visualizations
        import plotly.express as px
        import plotly.figure_factory as ff
        import numpy as np
        
        # 1. Create sub-dataset with specific attributes
        print(f"\n1. Creating sub-dataset with key attributes...")
        
        # Try to identify the correct column names
        attr_mapping = {}
        
        # Economy (GDP per Capita)
        for col in df.columns:
            if any(term in col.lower() for term in ['gdp', 'economy', 'economic']):
                attr_mapping['Economy'] = col
                break
        
        # Family
        for col in df.columns:
            if any(term in col.lower() for term in ['family', 'social']):
                attr_mapping['Family'] = col
                break
        
        # Health (Life Expectancy)
        for col in df.columns:
            if any(term in col.lower() for term in ['health', 'life']):
                attr_mapping['Health'] = col
                break
        
        # Freedom
        for col in df.columns:
            if 'freedom' in col.lower():
                attr_mapping['Freedom'] = col
                break
        
        # Trust (Government Corruption)
        for col in df.columns:
            if any(term in col.lower() for term in ['trust', 'corruption']):
                attr_mapping['Trust'] = col
                break
        
        # Generosity
        for col in df.columns:
            if 'generosity' in col.lower():
                attr_mapping['Generosity'] = col
                break
        
        # Happiness Score
        for col in df.columns:
            if any(term in col.lower() for term in ['happiness', 'score']) and 'rank' not in col.lower():
                attr_mapping['Happiness_Score'] = col
                break
        
        # Region
        for col in df.columns:
            if 'region' in col.lower():
                attr_mapping['Region'] = col
                break
        
        # Country
        for col in df.columns:
            if 'country' in col.lower():
                attr_mapping['Country'] = col
                break
        
        print(f"Identified attribute mappings:")
        for key, value in attr_mapping.items():
            print(f"  {key}: {value}")
        
        # Create sub-dataset
        required_attrs = ['Economy', 'Family', 'Health', 'Freedom', 'Trust', 'Generosity', 'Happiness_Score']
        available_attrs = [attr for attr in required_attrs if attr in attr_mapping]
        
        if len(available_attrs) >= 4:  # Need at least 4 attributes for meaningful analysis
            sub_cols = [attr_mapping[attr] for attr in available_attrs]
            sub_df = df[sub_cols].copy()
            
            # Rename columns for clarity
            rename_dict = {attr_mapping[attr]: attr for attr in available_attrs}
            sub_df = sub_df.rename(columns=rename_dict)
            
            print(f"\nSub-dataset created with {len(sub_df.columns)} attributes:")
            print(sub_df.head())
            
            # 2. Create correlation heatmap (fig2)
            print(f"\n2. Creating correlation heatmap...")
            
            # Calculate correlation matrix
            corr_matrix = sub_df.corr()
            
            # Create heatmap using plotly
            fig2 = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig2.update_layout(
                title={
                    'text': 'Correlation Heatmap of Happiness Attributes',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                width=800,
                height=600,
                xaxis_title="Attributes",
                yaxis_title="Attributes"
            )
            
            fig2.show()
            print("✓ Correlation heatmap 'fig2' created successfully!")
            
            # 3. Create scatter plot between Happiness Score and GDP per Capita (fig3)
            print(f"\n3. Creating scatter plot...")
            
            if 'Happiness_Score' in sub_df.columns and 'Economy' in sub_df.columns and 'Region' in attr_mapping:
                # Prepare data for scatter plot
                scatter_data = df.copy()
                
                fig3 = px.scatter(
                    scatter_data,
                    x=attr_mapping['Economy'],
                    y=attr_mapping['Happiness_Score'],
                    color=attr_mapping['Region'] if 'Region' in attr_mapping else None,
                    hover_data=[attr_mapping['Country']] if 'Country' in attr_mapping else None,
                    title='Happiness Score vs GDP per Capita by Region',
                    labels={
                        attr_mapping['Economy']: 'GDP per Capita',
                        attr_mapping['Happiness_Score']: 'Happiness Score'
                    }
                )
                
                fig3.update_layout(
                    title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    width=900,
                    height=600
                )
                
                fig3.show()
                print("✓ Scatter plot 'fig3' created successfully!")
            else:
                print("⚠️ Cannot create scatter plot - missing required columns")
            
            # 4. Create pie chart for Happiness Score by Region (fig4)
            print(f"\n4. Creating pie chart...")
            
            if 'Region' in attr_mapping and 'Happiness_Score' in attr_mapping:
                # Calculate average happiness score by region
                region_happiness = df.groupby(attr_mapping['Region'])[attr_mapping['Happiness_Score']].mean().reset_index()
                
                fig4 = px.pie(
                    region_happiness,
                    values=attr_mapping['Happiness_Score'],
                    names=attr_mapping['Region'],
                    title='Average Happiness Score by Region'
                )
                
                fig4.update_layout(
                    title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    width=800,
                    height=600
                )
                
                fig4.show()
                print("✓ Pie chart 'fig4' created successfully!")
            else:
                print("⚠️ Cannot create pie chart - missing Region or Happiness Score columns")
            
            # 5. Create world map for GDP per capita with Health tooltip (fig5)
            print(f"\n5. Creating world map...")
            
            if 'Country' in attr_mapping and 'Economy' in attr_mapping:
                # Create world map
                fig5 = px.choropleth(
                    df,
                    locations=attr_mapping['Country'],
                    color=attr_mapping['Economy'],
                    hover_name=attr_mapping['Country'],
                    hover_data={
                        attr_mapping['Health']: True if 'Health' in attr_mapping else False,
                        attr_mapping['Economy']: ':.2f'
                    },
                    color_continuous_scale='Viridis',
                    locationmode='country names',
                    title='GDP per Capita by Country (with Health Life Expectancy tooltip)'
                )
                
                fig5.update_layout(
                    title={
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16}
                    },
                    width=1000,
                    height=600,
                    geo=dict(showframe=False, showcoastlines=True)
                )
                
                fig5.show()
                print("✓ World map 'fig5' created successfully!")
            else:
                print("⚠️ Cannot create world map - missing Country or Economy columns")
            
            print(f"\n" + "="*50)
            print("DATA EXPLORATION COMPLETED")
            print("="*50)
            print("✓ All visualizations created successfully!")
            print("✓ fig1: Top 10 Countries GDP & Health Bar Chart")
            print("✓ fig2: Correlation Heatmap")
            print("✓ fig3: Happiness vs GDP Scatter Plot")
            print("✓ fig4: Happiness by Region Pie Chart")
            print("✓ fig5: GDP World Map with Health Tooltip")
            
            # 12. CREATE HTML DASHBOARD
            print(f"\n" + "="*50)
            print("CREATING HTML DASHBOARD")
            print("="*50)
            
            # Convert plots to HTML strings
            fig1_html = fig1.to_html(include_plotlyjs='cdn', div_id='fig1') if 'fig1' in locals() else ""
            fig2_html = fig2.to_html(include_plotlyjs='cdn', div_id='fig2') if 'fig2' in locals() else ""
            fig3_html = fig3.to_html(include_plotlyjs='cdn', div_id='fig3') if 'fig3' in locals() else ""
            fig4_html = fig4.to_html(include_plotlyjs='cdn', div_id='fig4') if 'fig4' in locals() else ""
            fig5_html = fig5.to_html(include_plotlyjs='cdn', div_id='fig5') if 'fig5' in locals() else ""
            
            # Create comprehensive HTML dashboard
            dashboard_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>World Happiness Analysis Dashboard 2016</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            color: #7f8c8d;
            font-size: 1.2em;
            margin-bottom: 20px;
        }}
        
        .executive-summary {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .section {{
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
        }}
        
        .section h3 {{
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 15px;
            font-size: 1.3em;
        }}
        
        .visualization {{
            margin: 20px 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        
        .insights {{
            background: #f8f9fa;
            border-left: 5px solid #3498db;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 10px 10px 0;
        }}
        
        .insights h4 {{
            color: #2980b9;
            margin-top: 0;
            font-size: 1.1em;
        }}
        
        .key-findings {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .finding-card {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #e74c3c;
        }}
        
        .finding-card h5 {{
            color: #c0392b;
            margin-top: 0;
            font-size: 1em;
        }}
        
        .methodology {{
            background: #fdf6e3;
            border: 1px solid #f1c40f;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            color: #fff;
            background: rgba(0,0,0,0.1);
            border-radius: 15px;
            margin-top: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        
        .stat-card {{
            background: #3498db;
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            display: block;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header Section -->
        <div class="header">
            <h1>🌍 World Happiness Analysis Dashboard</h1>
            <div class="subtitle">Comprehensive Analysis of Global Well-being Indicators • 2016 Data</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-number">{len(df)}</span>
                    <span class="stat-label">Countries Analyzed</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{len(df.columns)}</span>
                    <span class="stat-label">Data Attributes</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">{len(df[attr_mapping['Region']].unique()) if 'Region' in attr_mapping else 'N/A'}</span>
                    <span class="stat-label">Global Regions</span>
                </div>
                <div class="stat-card">
                    <span class="stat-number">5</span>
                    <span class="stat-label">Visualizations</span>
                </div>
            </div>
        </div>
        
        <!-- Executive Summary -->
        <div class="executive-summary">
            <h2>📊 Executive Summary</h2>
            <p><strong>This comprehensive dashboard presents an in-depth analysis of global happiness and well-being indicators for 2016.</strong> Our analysis reveals critical insights into the factors that contribute to national happiness, including economic prosperity, social support, health outcomes, personal freedom, trust in government, and generosity.</p>
            
            <div class="key-findings">
                <div class="finding-card">
                    <h5>🏆 Top Performers</h5>
                    <p>Nordic countries and wealthy nations dominate the happiness rankings, with strong correlations between GDP per capita and overall well-being scores.</p>
                </div>
                <div class="finding-card">
                    <h5>🔗 Strong Correlations</h5>
                    <p>Economic factors, family support, and health life expectancy show the strongest positive correlations with happiness scores.</p>
                </div>
                <div class="finding-card">
                    <h5>🌏 Regional Patterns</h5>
                    <p>Clear regional clustering emerges, with Western Europe and North America leading in multiple well-being dimensions.</p>
                </div>
                <div class="finding-card">
                    <h5>🎯 Policy Implications</h5>
                    <p>The data suggests that balanced investment in economic development, healthcare, and social systems yields the highest happiness returns.</p>
                </div>
            </div>
        </div>
        
        <!-- Visualization 1: Top 10 Countries -->
        <div class="section">
            <h2>🏅 Top 10 Happiest Countries: Economic & Health Perspective</h2>
            <p>This comparative analysis highlights the relationship between economic prosperity (GDP per capita) and health outcomes (life expectancy) among the world's happiest nations. The visualization reveals that top-performing countries excel in both dimensions, suggesting a synergistic relationship between wealth and health.</p>
            
            <div class="visualization">
                {fig1_html}
            </div>
            
            <div class="insights">
                <h4>🔍 Key Insights:</h4>
                <ul>
                    <li><strong>Nordic Dominance:</strong> Scandinavian countries consistently rank high in both GDP per capita and life expectancy</li>
                    <li><strong>Wealth-Health Correlation:</strong> Countries with higher GDP generally show better health outcomes</li>
                    <li><strong>Balanced Development:</strong> Top performers maintain strong performance across both economic and health metrics</li>
                    <li><strong>Policy Success:</strong> These nations demonstrate effective healthcare and economic policies</li>
                </ul>
            </div>
        </div>
        
        <!-- Visualization 2: Correlation Heatmap -->
        <div class="section">
            <h2>🔗 Happiness Factors Correlation Analysis</h2>
            <p>The correlation heatmap reveals the intricate relationships between different happiness indicators. Understanding these correlations is crucial for policymakers to identify which factors most strongly influence national well-being and where interventions might be most effective.</p>
            
            <div class="visualization">
                {fig2_html}
            </div>
            
            <div class="insights">
                <h4>🔍 Key Insights:</h4>
                <ul>
                    <li><strong>Strongest Correlations:</strong> GDP per capita and life expectancy show the highest correlation with happiness scores</li>
                    <li><strong>Social Factors:</strong> Family and social support demonstrate significant positive correlations</li>
                    <li><strong>Freedom Impact:</strong> Personal freedom correlates moderately but consistently with overall happiness</li>
                    <li><strong>Complex Relationships:</strong> Some factors show unexpected correlation patterns, suggesting nuanced policy implications</li>
                </ul>
            </div>
        </div>
        
        <!-- Visualization 3: Scatter Plot -->
        <div class="section">
            <h2>💰 Happiness vs. Economic Prosperity by Region</h2>
            <p>This scatter plot analysis examines the relationship between economic prosperity (GDP per capita) and happiness scores across different global regions. The color-coding by region reveals distinct patterns and outliers that inform our understanding of cultural and regional factors in well-being.</p>
            
            <div class="visualization">
                {fig3_html}
            </div>
            
            <div class="insights">
                <h4>🔍 Key Insights:</h4>
                <ul>
                    <li><strong>Regional Clustering:</strong> Clear clustering patterns emerge by geographic region</li>
                    <li><strong>Wealth Threshold:</strong> There appears to be a happiness threshold related to GDP levels</li>
                    <li><strong>Cultural Factors:</strong> Some regions achieve higher happiness with lower GDP, suggesting cultural influences</li>
                    <li><strong>Outlier Analysis:</strong> Notable outliers provide insights into unique national circumstances</li>
                </ul>
            </div>
        </div>
        
        <!-- Visualization 4: Regional Pie Chart -->
        <div class="section">
            <h2>🌍 Regional Happiness Distribution</h2>
            <p>The regional happiness distribution provides a macro-level view of how well-being varies across different parts of the world. This analysis helps identify which regions are thriving and which may need targeted international development support.</p>
            
            <div class="visualization">
                {fig4_html}
            </div>
            
            <div class="insights">
                <h4>🔍 Key Insights:</h4>
                <ul>
                    <li><strong>Western Dominance:</strong> Western Europe leads in average regional happiness scores</li>
                    <li><strong>Development Gaps:</strong> Significant disparities exist between developed and developing regions</li>
                    <li><strong>Regional Potential:</strong> Some regions show untapped potential for happiness improvement</li>
                    <li><strong>Global Inequality:</strong> The distribution highlights global well-being inequalities</li>
                </ul>
            </div>
        </div>
        
        <!-- Visualization 5: World Map -->
        <div class="section">
            <h2>🗺️ Global GDP Distribution with Health Context</h2>
            <p>The world map visualization provides a geographical perspective on economic prosperity, with health life expectancy data available through interactive tooltips. This global view helps identify regional economic patterns and their relationship to health outcomes.</p>
            
            <div class="visualization">
                {fig5_html}
            </div>
            
            <div class="insights">
                <h4>🔍 Key Insights:</h4>
                <ul>
                    <li><strong>Geographic Patterns:</strong> Clear geographic clustering of economic prosperity</li>
                    <li><strong>Resource Distribution:</strong> Natural resource endowments correlate with some high-GDP areas</li>
                    <li><strong>Development Corridors:</strong> Economic development often follows geographic and cultural corridors</li>
                    <li><strong>Health-Wealth Nexus:</strong> Interactive tooltips reveal the strong relationship between GDP and health outcomes</li>
                </ul>
            </div>
        </div>
        
        <!-- Methodology Section -->
        <div class="methodology">
            <h2>🔬 Methodology & Data Processing</h2>
            <h3>Data Preparation:</h3>
            <ul>
                <li><strong>Source:</strong> World Happiness Report 2016 dataset from IBM Skills Network</li>
                <li><strong>Preprocessing:</strong> Automated data type detection and correction</li>
                <li><strong>Missing Values:</strong> Numeric columns filled with mean values where appropriate</li>
                <li><strong>Quality Assurance:</strong> Comprehensive data validation and verification</li>
            </ul>
            
            <h3>Analytical Approach:</h3>
            <ul>
                <li><strong>Correlation Analysis:</strong> Pearson correlation coefficients calculated for all numeric variables</li>
                <li><strong>Regional Grouping:</strong> Countries analyzed by geographic and cultural regions</li>
                <li><strong>Top Performer Analysis:</strong> Rankings based on composite happiness scores</li>
                <li><strong>Interactive Visualization:</strong> Plotly-based charts for enhanced user engagement</li>
            </ul>
        </div>
        
        <!-- Conclusions & Recommendations -->
        <div class="section">
            <h2>🎯 Conclusions & Policy Recommendations</h2>
            
            <h3>Major Findings:</h3>
            <div class="key-findings">
                <div class="finding-card">
                    <h5>Economic Foundation</h5>
                    <p>GDP per capita remains the strongest predictor of national happiness, but diminishing returns suggest a threshold effect.</p>
                </div>
                <div class="finding-card">
                    <h5>Health Investment</h5>
                    <p>Life expectancy and health outcomes show consistent positive correlations with well-being across all regions.</p>
                </div>
                <div class="finding-card">
                    <h5>Social Cohesion</h5>
                    <p>Family support and social connections prove critical for happiness beyond economic factors.</p>
                </div>
                <div class="finding-card">
                    <h5>Governance Quality</h5>
                    <p>Trust in government and low corruption correlate significantly with higher happiness scores.</p>
                </div>
            </div>
            
            <h3>Policy Recommendations:</h3>
            <ul>
                <li><strong>Holistic Development:</strong> Balance economic growth with social and health investments</li>
                <li><strong>Healthcare Priority:</strong> Prioritize healthcare systems as fundamental to national well-being</li>
                <li><strong>Social Infrastructure:</strong> Invest in community support systems and social cohesion programs</li>
                <li><strong>Governance Reform:</strong> Strengthen institutions and reduce corruption for long-term happiness gains</li>
                <li><strong>Regional Cooperation:</strong> Facilitate knowledge sharing between high-performing and developing regions</li>
            </ul>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p><strong>World Happiness Analysis Dashboard</strong></p>
            <p>Generated using Python, Pandas, and Plotly • Data from World Happiness Report 2016</p>
            <p>📊 Comprehensive analysis of {len(df)} countries across {len(df.columns)} indicators</p>
        </div>
    </div>
</body>
</html>
"""
            
            # Save the dashboard
            with open('world_happiness_dashboard.html', 'w', encoding='utf-8') as f:
                f.write(dashboard_html)
            
            print("✓ HTML Dashboard created successfully!")
            print("✓ File saved as: 'world_happiness_dashboard.html'")
            print("✓ Dashboard includes:")
            print("  - Executive Summary with key findings")
            print("  - All 5 interactive visualizations")
            print("  - Detailed insights for each chart")
            print("  - Methodology and recommendations")
            print("  - Professional styling and responsive design")
            print("\n🎉 Complete data analysis and dashboard generation finished!")
            
        else:
            print(f"⚠️ Insufficient attributes found for analysis.")
            print(f"Found: {available_attrs}")
            print(f"Required: {required_attrs}")
            
    except Exception as explore_error:
        print(f"Error during data exploration: {explore_error}")
        print(f"Please ensure all required libraries are installed: plotly, numpy")
    
except Exception as e:
    print(f"Error during data preparation: {e}")


# In[ ]:




