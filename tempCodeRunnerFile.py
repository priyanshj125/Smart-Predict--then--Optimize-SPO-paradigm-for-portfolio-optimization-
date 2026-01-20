 if isinstance(data.columns,pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                df=data['Close']
            else:
                df=data
        else:
            df=data['Close']  
        df=df.ffill().dropna()
        print("download complete")
