import React, { Component,  Fragment} from 'react';

import {MuiThemeProvider, Toolbar, ToolbarTitle} from 'material-ui';
import Options from './Options.js';
import Modified from './Modified.js';
import Help from './Help.js';
import {indigo500, red800} from 'material-ui/styles/colors';
import getMuiTheme from 'material-ui/styles/getMuiTheme';

import './App.css';
const muiTheme = getMuiTheme({
  palette: {
    primary1Color: indigo500,
    accent1Color: red800
  },
});

class App extends Component {
  constructor(props) {
    super(props);
    
    this.state = {
      netLoaded: true,
      image: 0,
      brushSize: 15,
      reset: 0,
      image_type: 'celebahq',
      mask_mode: 'gconv',
      random: 0,
      eraserEnable: 0,
      loadin:false,
      download:0
    };

  }

  imageTypeChanged = (e) => {
    
    this.setState({
      image_type: e.target.value
    });
  }
  maskModeChanged = (e) => {
    this.setState({
      mask_mode:  e.target.value
    });
  }

  eraserChanged = (event, eraserEnable) => {
    this.setState({eraserEnable: eraserEnable});
  };

imageUpload = (e) => {
    this.setState({
      image: 1
    });
  }


  brushChanged = (e, val) => {
    this.setState({
      brushSize: val
    });
  }
  
  reset = (e) => {
    this.setState({
      reset: 1
    }, function() {
      this.setState({
        reset: 0
      });
    });
  }
  download = (e) => {
    this.setState({
      download: 1
    }, function() {
      this.setState({
        download: 0
      });
    });
  }
  randomImage = (e) => {
    this.setState({
      random: 1
    }, function() {
      this.setState({
        random: 0
      });
    });
  }
  

  render() {
    if (this.state.netLoaded) {
      return (
          // <Fragment>
          <MuiThemeProvider muiTheme={muiTheme}>
            <div id="mui-container">
              <div id="toolbar">
            <Toolbar style={{backgroundColor: "#7db6bf", color: "#1e1e1e", paddingTop:1, paddingBottom: 1}}>
              <a href="/"><ToolbarTitle text="交互式图像修复" /></a>
              {/* <img src='logo_buaa.png'></img> */}
            </Toolbar>
            </div>
            <div id="main">
            <Modified  image_type={this.state.image_type}  mask_mode={this.state.mask_mode} brushSize={this.state.brushSize} reset={this.state.reset} download={this.state.download}
                  random={this.state.random}  eraserEnable={this.state.eraserEnable}/>
            <Options imageTypeChanged={this.imageTypeChanged}  imageUpload={this.imageUpload} maskModeChanged={this.maskModeChanged}
            brushChanged={this.brushChanged}  reset={this.reset} randomImage={this.randomImage} random={this.state.random} download={this.download} eraserChanged={this.eraserChanged}
              brushSize={this.state.brushSize} image_type={this.state.image_type} mask_mode={this.state.mask_mode}/>
          {/* </Fragment> */}
          </div>
          </div>
          </MuiThemeProvider>
      );
    } else {
      return (
        <h3 id="loading-text">加载出错</h3>
      );
    }
  }
}

export default App;
