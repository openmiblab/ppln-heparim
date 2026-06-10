# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 13:55:39 2026

@author: md1jbree
"""
import pandas as pd

def reformat_dice_to_comparison(input_csv, output_csv):
    # 1. Load the raw global dice data
    df = pd.read_csv(input_csv)
    
    # 2. Pivot the table: 
    # Index = Label (Segment Name)
    # Columns = Case (Mohammed vs AI comparisons)
    # Values = Dice Score
    comparison_df = df.pivot(index='Patient_ID', columns='Manual_Segment', values='Dice_Score')
    
    # 3. Add an "Average" column to see overall performance per segment
    comparison_df['Mean_Dice'] = comparison_df.mean(axis=1)
    
    # 4. Save the reformatted version
    comparison_df.to_csv(output_csv)
    
    print(f"✅ Comparison CSV saved to: {output_csv}")
    print(comparison_df.head())
    

reformat_dice_to_comparison("C:\\Users\\md1jbree\\output\\batch_processing\\global_dice_summary.csv", "C:\\Users\\md1jbree\\output\\batch_processing\\global_dice_summary_reformat.csv")