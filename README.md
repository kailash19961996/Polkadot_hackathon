# Polkadot Transaction Analysis Tool

### AI-Powered

## Overview

This application is designed to analyze transactions on the blockchain network, helping to identify suspicious activities and potential outliers using machine learning and AI. It was developed as part of the Polkadot x EasyA London Hackathon.

[Click here to try the app](https://polkadot-app2-715cfeb3743d.herokuapp.com/)

## Features

- **Customizable Data Processing**: Users can choose the frequency and volume of data to process.
- **Graphical Visualization**: Displays suspicious senders and receivers based on transaction patterns.
- **Machine Learning Integration**: Utilizes Isolation Forest algorithm to detect outliers.
- **Blacklist Checking**: Identified outliers are cross-referenced with blacklisted accounts.
- **Interactive Graphs**: 2D and 3D graphs for visualizing transaction patterns.
- **Detailed Information**: Hover over nodes or connections to view transaction details.

## How It Works

1. **Data Selection**: Users specify the amount and frequency of data to analyze.
2. **Initial Analysis**: The system calculates average transaction values to identify potential suspicious activities.
3. **Machine Learning Detection**: An Isolation Forest algorithm is applied to spot outliers, which are marked as red X's on the graph.
4. **Blacklist Verification**: Outliers are checked against a list of blacklisted accounts.
5. **Visualization**: All outliers are displayed on 2D and 3D graphs for pattern analysis.
6. **Detailed Inspection**: Users can hover over graph elements to view more transaction details.

## Future Enhancements

- Implement more complex rules for identifying suspicious transactions.
- Expand machine learning capabilities for more accurate outlier detection.
- Integrate additional data sources for comprehensive analysis.


