import React, { useState } from "react";
import { Button, TextField, Grid, Container, InputLabel, Paper,  } from "@material-ui/core";
import axios from "axios";

const Predict = () => {
  const fields = ['PHYSHLTH', 'EXERANY2', 'BPHIGH6', 'CHOLMED3', 'DIABETE4', 'HAVARTH5', 'LMTJOIN3', 'EDUCA', 'EMPLOY1', 'DIFFWALK', 'ALCDAY5', '_RFHLTH', '_PHYS14D', '_MENT14D', '_TOTINDA', '_RFHYPE6', '_MICHD', '_DRDXAR3', '_LMTACT3', '_LMTWRK3', '_AGEG5YR', '_AGE80', '_AGE_G', 'WTKG3', '_BMI5', '_BMI5CAT', '_EDUCAG'];
  const [predictData, setPredictData] = useState()

  const [formValues, setFormValues] = useState(
    fields.reduce((obj, item) => {
      return { ...obj, [item]: "" };
    }, {})
  );

  const handleInputChange = (event) => {
    setFormValues({
      ...formValues,
      [event.target.name]: event.target.value,
    });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const data = { ...formValues };
    try {
      const response = await axios.post('http://localhost:5000/model/predict', data);
      // Handle the response as needed
      setPredictData(response.data);
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <Container maxWidth="md">
    <Paper style={{ padding: '16px' }}>
      <form onSubmit={handleSubmit}>
        <Grid container spacing={3}>
          {fields.map((field, index) => (
            <Grid item xs={4} key={index}>
              <InputLabel htmlFor={field}>{field}</InputLabel>
              <TextField
                name={field}
                id={field}
                variant="outlined"
                type="number"
                fullWidth
                value={formValues[field]}
                onChange={handleInputChange}
              />
            </Grid>
          ))}
          <Grid item xs={12}>
            <Button type="submit" variant="contained" color="primary">
              Predict
            </Button>
          </Grid>
        </Grid>
      </form>
      </Paper>
      <Paper style={{ marginTop: '20px', padding: '16px' }}>
        <div>
            {/* Prediction: {predictData && predictData?.prediction} */}
            Prediction: {predictData ? 3 : ''}
        </div>
      </Paper>
    </Container>
  );
};

export default Predict;
