import React from 'react';
import { Typography, Box, Container, Paper } from '@material-ui/core';

const About = () => (
  <Container>
    <Paper style={{ padding: '16px' }}>
        <Box my={4}>
        <Typography variant="h4" component="h1" gutterBottom>
            About Our Project
        </Typography>
        
        <Typography variant="body1" gutterBottom>
        The Behavioral Risk Factor Surveillance System (BRFSS) 2021 dataset is a valuable resource, providing
        comprehensive data that sheds light on critical health-related factors affecting the U.S. population. This
        dataset is produced annually by the Centers for Disease Control and Prevention (CDC) as part of its
        ongoing efforts to monitor health conditions and risk behaviors.
        </Typography>

        <Typography variant="body1" gutterBottom>
        In this experimental project, my goal is to predict the general health of patients using BRFSS 2021 dataset
with the help of Machine-Learning. Also, I will contribute a PUI to this project that helps in managing
the ML model easier through a web interface.
        </Typography>

        <Typography variant="body1" gutterBottom>
        My motivation for this project is to experiment with the things that I’ve learned through the course and
apply them to a real-world dataset, which is BRFSS 2021. Also, developing a PUI (Perceptual User Inter-
face) can be challenging by its nature. Because handling big data on user interfaces is often challenging
work since most of the user interface programs run on a client which has a low amount of computational
resources.
        </Typography>
        </Box>
    </Paper>
  </Container>
);

export default About;
