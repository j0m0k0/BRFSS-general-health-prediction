import React, { useEffect, useState } from 'react';
import { Box, Typography, Grid, Paper } from '@material-ui/core';
import { Pie } from 'react-chartjs-2';
import 'chart.js/auto';


const Overview = () => {
    const [modelData, setModelData] = useState(null);

    useEffect(() => {
        // Fetch data from server
        // This is a placeholder. Replace with your API endpoint
        fetch("http://localhost:5000/model/info")
            .then(response => response.json())
            .then(data => setModelData(data))
            .catch(error => console.error('Error:', error));
    }, []);

    const pieData = {
        labels: ['Train Samples', 'Evaluation Samples', 'Test Samples'],
        datasets: [
            {
                data: [modelData?.train_samples, modelData?.evaluation_samples, modelData?.test_samples],
                backgroundColor: ['red', 'blue', 'green'],
            },            
        ],
        width: '10px'
    };

    return (        
        <div>            
            <Grid container alignItems='center' component={Paper} style={{padding: '20px', margin: '20px'}}>                
                <Grid item xs={12}>
                    <Box mb={4}>
                    <Typography variant="h4">Overview</Typography>
                </Box>
                </Grid>
                <Grid item xs={6}>
                    <Box p={2} boxShadow={2}>
                        <Typography variant="h6">Training Details</Typography>
                        <Typography variant="body1">Train Samples: {modelData?.train_samples}</Typography>
                        <Typography variant="body1">Evaluation Samples: {modelData?.evaluation_samples}</Typography>
                        <Typography variant="body1">Test Samples: {modelData?.test_samples}</Typography>
                    </Box>
                </Grid>
                <Grid item xs={6}>
                    <Box p={2} boxShadow={2}>
                        <div style={{height: '300px', width: '300px'}}>
                            <Pie data={pieData} />
                        </div>
                    </Box>
                </Grid>   
            </Grid>

            <Grid container alignItems='center' component={Paper} style={{padding: '20px', margin: '20px'}}>                
                    <Grid item xs={12}>
                        <Box p={2} boxShadow={2}>
                            <Typography variant="h6">Algorithm Details</Typography>
                            <Typography variant="body1">Name: {modelData?.model_name}</Typography>
                            {modelData && Object.keys(modelData?.hyperparameters).map((k, index) => (
                                <Typography key={index} variant="body1">{k}: {modelData?.hyperparameters[k]}</Typography>
                            ))}
                        </Box>
                    </Grid>
            </Grid>
        </div>
    );
}

export default Overview;
