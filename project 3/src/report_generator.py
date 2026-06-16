import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from datetime import datetime

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def export_csv(forecast_df: pd.DataFrame) -> str:
    """
    Export forecast results to CSV format string.
    """
    output_df = forecast_df.copy()
    # If the index is not named Date, set it
    if output_df.index.name is None:
        output_df.index.name = 'Date'
        
    return output_df.to_csv(index=True)

def export_excel(df_history: pd.DataFrame, test_forecast_df: pd.DataFrame, 
                 future_forecast_df: pd.DataFrame, evaluation_metrics: dict, 
                 arima_order: tuple, ticker: str = "Stock") -> bytes:
    """
    Export forecasts, metrics, and history summary to an Excel file with multiple sheets.
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Future Predictions
        future_export = future_forecast_df.copy()
        future_export.index = future_export.index.strftime('%Y-%m-%d')
        future_export.to_excel(writer, sheet_name='Future Forecast')
        
        # Sheet 2: Test Set Comparison (Actual vs Predicted)
        if test_forecast_df is not None and not test_forecast_df.empty:
            test_export = test_forecast_df.copy()
            test_export.index = test_export.index.strftime('%Y-%m-%d')
            test_export.to_excel(writer, sheet_name='Test Forecast Evaluation')
            
        # Sheet 3: Evaluation Metrics
        metrics_df = pd.DataFrame([
            {'Metric': 'Best ARIMA Order', 'Value': f"ARIMA{arima_order}"},
            {'Metric': 'Mean Absolute Error (MAE)', 'Value': evaluation_metrics.get('MAE', np.nan)},
            {'Metric': 'Mean Squared Error (MSE)', 'Value': evaluation_metrics.get('MSE', np.nan)},
            {'Metric': 'Root Mean Squared Error (RMSE)', 'Value': evaluation_metrics.get('RMSE', np.nan)},
            {'Metric': 'Mean Absolute Percentage Error (MAPE %)', 'Value': evaluation_metrics.get('MAPE', np.nan)},
            {'Metric': 'R-squared (R²)', 'Value': evaluation_metrics.get('R2', np.nan)},
            {'Metric': 'Forecast Accuracy (%)', 'Value': evaluation_metrics.get('Accuracy_Pct', np.nan)},
            {'Metric': 'Accuracy Rating', 'Value': evaluation_metrics.get('Rating', 'N/A')},
        ])
        metrics_df.to_excel(writer, sheet_name='Model Performance', index=False)
        
        # Sheet 4: Historical Stats Summary
        hist_stats = df_history['Close'].describe().reset_index()
        hist_stats.columns = ['Stat', 'Value']
        hist_stats.to_excel(writer, sheet_name='Historical Summary', index=False)
        
    return output.getvalue()

def generate_business_insights(ticker: str, metrics: dict, future_df: pd.DataFrame, 
                               ma_signals: pd.DataFrame) -> list[str]:
    """
    Generate automated business insights text based on forecasting outputs and moving average trends.
    """
    insights = []
    
    # 1. Prediction direction
    start_price = future_df['Forecast'].iloc[0]
    end_price = future_df['Forecast'].iloc[-1]
    pct_change = ((end_price - start_price) / start_price) * 100
    days = len(future_df)
    
    direction = "upward" if pct_change > 0 else "downward"
    strength = "strong" if abs(pct_change) > 5 else "moderate"
    
    insights.append(
        f"The model forecasts a {strength} {direction} trend for {ticker} over the next {days} days, "
        f"with closing prices projected to move from {start_price:,.2f} to {end_price:,.2f} "
        f"(a change of {pct_change:.2f}%)."
    )
    
    # 2. Risk / Uncertainty indicator
    avg_ci_width = (future_df['Upper_CI'] - future_df['Lower_CI']).mean()
    pct_ci_width = (avg_ci_width / future_df['Forecast'].mean()) * 100
    
    if pct_ci_width < 10:
        insights.append(
            f"The 95% confidence intervals are relatively narrow (average width of {pct_ci_width:.1f}% of price), "
            f"suggesting lower statistical uncertainty and high stability in this asset's current price trajectory."
        )
    else:
        insights.append(
            f"The confidence bounds span a wider range (average width of {pct_ci_width:.1f}% of price). "
            f"This indicates significant volatility and forecast uncertainty. Traders should exercise caution "
            f"and implement tight stop-losses if acting on this prediction."
        )
        
    # 3. Moving Average Signals Check
    if ma_signals is not None and not ma_signals.empty:
        latest_signal = ma_signals.iloc[0]
        date_str = latest_signal['Date'].strftime('%Y-%m-%d')
        insights.append(
            f"The latest moving average crossover event was a **{latest_signal['Signal']}** signal trigger "
            f"({latest_signal['Type']}) on {date_str} at a price of {latest_signal['Price']:,.2f}. "
            f"This indicates a potential shift or continuation in the underlying momentum."
        )
    else:
        insights.append(
            "No recent moving average crossover signals (like Golden or Death Crosses) were detected, "
            "indicating that the stock is currently trading in a consolidated or range-bound trend."
        )
        
    # 4. Actionable recommendation
    rating = metrics.get('Rating', 'Reasonable')
    if direction == "upward" and rating in ["Highly Accurate", "Good"]:
        insights.append(
            f"**Recommendation:** Based on a '{rating}' model with an upward forecast, {ticker} presents "
            f"a potential buying opportunity. Long positions can be scaled in, keeping the lower confidence bound "
            f"({future_df['Lower_CI'].iloc[-1]:,.2f} at day {days}) in mind as a worst-case risk floor."
        )
    elif direction == "downward" and rating in ["Highly Accurate", "Good"]:
        insights.append(
            f"**Recommendation:** The forecast indicates a downward trend. Since the model has a '{rating}' historical "
            f"accuracy, it is advisable to hedge current positions, defer new long entries, or consider short opportunities "
            f"if trading derivative instruments."
        )
    else:
        insights.append(
            "**Recommendation:** Due to intermediate accuracy or high volatility bounds, it is recommended to "
            "use this forecast in conjunction with secondary momentum indicators (e.g., RSI, MACD) rather than as a "
            "stand-alone trading trigger."
        )
        
    return insights

def export_pdf_report(df_history: pd.DataFrame, test_forecast_df: pd.DataFrame, 
                      future_forecast_df: pd.DataFrame, evaluation_metrics: dict, 
                      arima_order: tuple, stationarity_info: dict, ma_signals: pd.DataFrame,
                      ticker: str = "Stock") -> bytes:
    """
    Generate a highly styled, professional PDF report of stock forecasting.
    Includes in-memory charts and business insights.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    # Setup styles
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor('#1A365D') # Deep Navy
    secondary_color = colors.HexColor('#2B6CB0') # Slate Blue
    accent_color = colors.HexColor('#E2E8F0') # Light gray
    text_color = colors.HexColor('#2D3748') # Dark Charcoal
    success_color = colors.HexColor('#2F855A') # Forest Green
    
    # Custom paragraph styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=primary_color,
        spaceAfter=15,
        alignment=TA_CENTER
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=15,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=8,
        borderPadding=(0, 0, 2, 0),
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=8,
        alignment=TA_LEFT
    )
    
    insight_style = ParagraphStyle(
        'DocInsight',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=text_color,
        leading=14,
        spaceAfter=6,
        leftIndent=15,
        firstLineIndent=-10
    )
    
    meta_style = ParagraphStyle(
        'DocMeta',
        fontName='Helvetica-Oblique',
        fontSize=9,
        textColor=colors.HexColor('#718096'),
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    # 1. Header Section
    story.append(Paragraph(f"Stock Price Forecasting & Technical Report", title_style))
    story.append(Paragraph(f"Ticker Asset: <b>{ticker}</b> &bull; Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", meta_style))
    story.append(Spacer(1, 10))
    
    # 2. Executive Statistics Summary
    story.append(Paragraph("1. Historical Statistics Summary", h1_style))
    latest_price = df_history['Close'].iloc[-1]
    avg_price = df_history['Close'].mean()
    max_price = df_history['Close'].max()
    min_price = df_history['Close'].min()
    
    stats_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Latest Close Price", f"{latest_price:,.2f}", "Average Price", f"{avg_price:,.2f}"],
        ["Highest Price", f"{max_price:,.2f}", "Lowest Price", f"{min_price:,.2f}"],
        ["Data Points (Days)", f"{len(df_history)}", "Date Range", f"{df_history.index.min().strftime('%Y-%m-%d')} to {df_history.index.max().strftime('%Y-%m-%d')}"]
    ]
    
    stats_table = Table(stats_data, colWidths=[130, 130, 130, 130])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 12))
    
    # 3. Model Configuration & Stationarity Testing
    story.append(Paragraph("2. Stationarity & ARIMA Configuration", h1_style))
    
    adf_p = stationarity_info.get('p_value', np.nan)
    adf_stat = stationarity_info.get('test_statistic', np.nan)
    is_stat = "Stationary" if stationarity_info.get('is_stationary', False) else "Non-Stationary"
    
    model_data = [
        ["Parameter / Test", "Result / Configuration", "Details"],
        ["ADF Test Statistic", f"{adf_stat:.4f}", f"p-value: {adf_p:.4f} ({is_stat})"],
        ["Best ARIMA Order", f"ARIMA{arima_order}", "Determined via AIC/BIC auto search"],
        ["Forecast Window", f"{len(future_forecast_df)} Days", "Out-of-sample forward projections"]
    ]
    
    model_table = Table(model_data, colWidths=[150, 170, 200])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 12))
    
    # 4. Model Forecast Evaluation (Test Split)
    story.append(Paragraph("3. Model Forecasting Accuracy (Test Validation)", h1_style))
    
    mae = evaluation_metrics.get('MAE', np.nan)
    rmse = evaluation_metrics.get('RMSE', np.nan)
    mape = evaluation_metrics.get('MAPE', np.nan)
    r2 = evaluation_metrics.get('R2', np.nan)
    rating = evaluation_metrics.get('Rating', 'N/A')
    
    eval_data = [
        ["Metric", "Value", "Standard Interpretation"],
        ["Mean Absolute Error (MAE)", f"{mae:.4f}" if not np.isnan(mae) else "N/A", "Average absolute error magnitude"],
        ["Root Mean Squared Error (RMSE)", f"{rmse:.4f}" if not np.isnan(rmse) else "N/A", "Penalizes larger outliers in residuals"],
        ["Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%" if not np.isnan(mape) else "N/A", "Average percentage prediction variance"],
        ["R-squared (R²)", f"{r2:.4f}" if not np.isnan(r2) else "N/A", "Proportion of variance explained by model"],
        ["Accuracy Rating", f"{rating}", evaluation_metrics.get('Interpretation', '')]
    ]
    
    eval_table = Table(eval_data, colWidths=[180, 100, 240])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('TEXTCOLOR', (1,4), (1,4), success_color if rating in ["Highly Accurate", "Good"] else colors.orange),
        ('FONTNAME', (1,4), (1,4), 'Helvetica-Bold'),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 15))
    
    # Let's add the future predictions plot
    story.append(Paragraph("4. Visual Forecast & Confidence Intervals", h1_style))
    
    # We will generate a matplotlib plot for the PDF, save to memory, and insert as image
    try:
        plt.figure(figsize=(10, 4))
        # Plot last 60 points of history
        hist_plot = df_history['Close'].iloc[-60:]
        plt.plot(hist_plot.index, hist_plot.values, label='Historical Price', color='#2b6cb0', linewidth=2)
        
        # Plot future forecast
        plt.plot(future_forecast_df.index, future_forecast_df['Forecast'].values, label='ARIMA Forecast', color='#2f855a', linewidth=2)
        # Plot confidence intervals
        plt.fill_between(future_forecast_df.index, 
                         future_forecast_df['Lower_CI'].values, 
                         future_forecast_df['Upper_CI'].values, 
                         color='#2f855a', alpha=0.15, label='95% Confidence Interval')
                         
        plt.title(f"{ticker} Historical Prices & Future Predictions", fontsize=12, fontweight='bold', color='#1a365d')
        plt.xlabel("Date", fontsize=9)
        plt.ylabel("Closing Price", fontsize=9)
        plt.legend(loc='upper left', fontsize=8)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=15, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=200)
        chart_buffer.seek(0)
        plt.close()
        
        img = Image(chart_buffer, width=480, height=192)
        story.append(img)
    except Exception as e:
        story.append(Paragraph(f"<i>Error generating chart: {str(e)}</i>", body_style))
        
    story.append(Spacer(1, 12))
    
    # 5. Business Insights Page/Section
    story.append(Paragraph("5. AI-Generated Business Insights & Recommendations", h1_style))
    insights = generate_business_insights(ticker, evaluation_metrics, future_forecast_df, ma_signals)
    for ins in insights:
        story.append(Paragraph(f"&bull; {ins}", insight_style))
        
    story.append(Spacer(1, 12))
    
    # 6. Future Predictions Data Summary
    story.append(Paragraph("6. Future Forecast Price List", h1_style))
    
    # Build preview of future forecast (e.g. up to 10 rows to fit page, otherwise summary)
    future_preview_df = future_forecast_df.copy()
    if len(future_preview_df) > 10:
        # Show first 5 and last 5 with a separator row
        f5 = future_preview_df.head(5)
        l5 = future_preview_df.tail(5)
        
        f_table_data = [["Date", "Forecasted Price", "95% Lower Bound", "95% Upper Bound"]]
        for d, row in f5.iterrows():
            f_table_data.append([d.strftime('%Y-%m-%d'), f"{row['Forecast']:,.2f}", f"{row['Lower_CI']:,.2f}", f"{row['Upper_CI']:,.2f}"])
            
        f_table_data.append(["...", "...", "...", "..."])
        
        for d, row in l5.iterrows():
            f_table_data.append([d.strftime('%Y-%m-%d'), f"{row['Forecast']:,.2f}", f"{row['Lower_CI']:,.2f}", f"{row['Upper_CI']:,.2f}"])
    else:
        f_table_data = [["Date", "Forecasted Price", "95% Lower Bound", "95% Upper Bound"]]
        for d, row in future_preview_df.iterrows():
            f_table_data.append([d.strftime('%Y-%m-%d'), f"{row['Forecast']:,.2f}", f"{row['Lower_CI']:,.2f}", f"{row['Upper_CI']:,.2f}"])
            
    future_table = Table(f_table_data, colWidths=[130, 130, 130, 130])
    future_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F7FAFC')]),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
    ]))
    story.append(future_table)
    
    # Build Document
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
