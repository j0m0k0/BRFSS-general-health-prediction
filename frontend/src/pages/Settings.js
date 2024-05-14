import React, { useState } from 'react';
import axios from 'axios';
import { Button, TextField, Grid, Paper } from '@material-ui/core';

const Settings = () => {
  const [state, setState] = useState({
    eval_percent: '',
    test_percent: '',
    train_percent: '',
    max_features: '',
    min_sample_splits: '',
    n_estimators: '',
  });

  const handleChange = (event) => {
    setState({ ...state, [event.target.name]: event.target.value });
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    const data = {
        eval_percent: parseFloat(state.eval_percent),
        test_percent: parseFloat(state.test_percent),
        train_percent: parseFloat(state.train_percent),
        max_features: state.max_features,
        min_sample_splits: parseInt(state.min_sample_splits, 10),
        n_estimators: parseInt(state.n_estimators, 10),
      };

    axios.post('http://localhost:5000/model/hyperparameters', data)
      .then(response => {
        console.log(response.data);
      })
      .catch(error => {
        console.log(error);
      });
  };

  return (
    <Grid container justify="center">
      <Grid item xs={12} sm={8} md={6}>
        <Paper style={{ padding: '16px' }}>
          <form onSubmit={handleSubmit}>
            <TextField
              name="eval_percent"
              label="Evaluation Percent"
              type="number"
              value={state.eval_percent}
              onChange={handleChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="test_percent"
              label="Test Percent"
              type="number"
              value={state.test_percent}
              onChange={handleChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="train_percent"
              label="Train Percent"
              type="number"
              value={state.train_percent}
              onChange={handleChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="max_features"
              label="Max Features"
              value={state.max_features}
              onChange={handleChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="min_sample_splits"
              label="Minimum Sample Splits"
              type="number"
              value={state.min_sample_splits}
              onChange={handleChange}
              fullWidth
              margin="normal"
            />
            <TextField
              name="n_estimators"
              label="Number of Estimators"
              type="number"
              value={state.n_estimators}
              onChange={handleChange}
              fullWidth
              margin="normal"
            />
            <Button type="submit" variant="contained" color="primary" style={{ marginTop: '16px' }}>
              Submit
            </Button>
          </form>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default Settings;
