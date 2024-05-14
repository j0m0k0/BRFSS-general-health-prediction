import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { Drawer, List, ListItem, ListItemText, ListItemIcon, AppBar, Toolbar, Typography } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';

// Import your page components
import Overview from './pages/Overview';
import Predict from './pages/Predict';
import Dataset from './pages/Dataset';
import Settings from './pages/Settings';
import About from './pages/About';

const drawerWidth = 240;

const useStyles = makeStyles((theme) => ({
  root: {
    display: 'flex',
  },
  appBar: {
    zIndex: theme.zIndex.drawer + 1,
  },
  drawer: {
    width: drawerWidth,
    flexShrink: 0,
  },
  drawerPaper: {
    width: drawerWidth,
  },
  content: {
    flexGrow: 1,
    padding: theme.spacing(3),
  },
}));

function ListItemLink({ icon, primary, to }) {
  const location = useLocation();  // Get the current route

  const renderLink = React.useMemo(
    () => React.forwardRef((itemProps, ref) => <Link to={to} ref={ref} {...itemProps} />),
    [to],
  );

  return (
    <li>
      <ListItem button component={renderLink} selected={location.pathname === to}>
        {icon ? <ListItemIcon>{icon}</ListItemIcon> : null}
        <ListItemText primary={primary} />
      </ListItem>
    </li>
  );
}

const App = () => {
  const classes = useStyles();

  return (
    <Router>
      <div className={classes.root}>
        <AppBar position="fixed" className={classes.appBar}>
          <Toolbar>
            <Typography variant="h6" noWrap>
              General Health AI Model Control Panel
            </Typography>
          </Toolbar>
        </AppBar>
        <Drawer
          className={classes.drawer}
          variant="permanent"
          classes={{ paper: classes.drawerPaper }}
        >
          <Toolbar />
          <List>
                    <ListItemLink to="/" primary="Overview" />
                    <ListItemLink to="/predict" primary="Predict" />
                    <ListItemLink to="/dataset" primary="Dataset" />
                    <ListItemLink to="/settings" primary="Settings" />
                    <ListItemLink to="/about" primary="About" />
                </List>
        </Drawer>
        <main className={classes.content}>
          <Toolbar />
          <Routes>
            <Route exact path="/" element={<Overview />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/dataset" element={<Dataset />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
