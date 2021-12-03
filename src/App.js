import React, { Component } from 'react';

// import {SqueezeNet} from './squeezenet/squeezenet.js';
import {MuiThemeProvider, Toolbar, ToolbarTitle} from 'material-ui';
import {indigo500, red800} from 'material-ui/styles/colors';
import getMuiTheme from 'material-ui/styles/getMuiTheme';

import Box from '@material-ui/core/Box';

import Options from './Options.js';
import Modified from './Modified.js';
import Help from './Help.js';


import './App.css';

function Item(props) {
  const { sx, ...other } = props;
  return (
    <Box
      sx={{
        p: 0,
        m: 0,
        borderRadius: 1,
        fontSize: '1rem',
        fontWeight: '700',
        ...sx,
      }}
      {...other}
    />
  );
}
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
      image: '普通.jpg',
      topK: new Map(),
      brushSize: 15,
      blurSize: 2,
      blur: 0,
      reset: 0,
      processedImage: null,
      processedFinished: 0,
      model: 'gconv',
      random: 0,
      eraserEnable: 0
    };

  }

  imageChanged = (e) => {
    
    this.setState({
      image: e.target.value
    });
  }
  modelChanged = (e) => {
    this.setState({
      model:  e.target.value
    });
  }

  eraserChanged = (event, eraserEnable) => {
    this.setState({eraserEnable: eraserEnable});
  };

  // 文件上传没完成！！
  imageUpload = (e) => {
    e.preventDefault()
    let file = e.target
    var reader = new FileReader();
    reader.readAsDataURL(file.files[0].getNative());
    const img = new Image(256, 256);
    reader.onload = function (e) {
        img.src = this.result;
    }
  }

  imageProcessed = (e) => {
    this.setState({
      processedImage: e,
      processedFinished: 1
    }, function() {
      this.setState({
        processedFinished: 0
      });
    });
  }

  imageRandomed = (e) => {
    console.log('imageRandomed:'+e)

    this.setState({
      image: e
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
        <MuiThemeProvider muiTheme={muiTheme}>
          <div id="mui-container">
            <Toolbar id="header" style={{backgroundColor: "rgba(63, 81, 181,1.0)", color: "white"}}>
              <a href="/"><ToolbarTitle text="交互式人脸修复" /></a>
            </Toolbar>
            <div id="main">
          <Box sx={{
            display: 'flex',
            flexDirection: 'row',
            p: 0,
            m: 0,
          }}>
            <Item sx={{flexShrink: 1}}>
                <Options imageChanged={this.imageChanged}  imageUpload={this.imageUpload} modelChanged={this.modelChanged} 
                brushChanged={this.brushChanged}  reset={this.reset} randomImage={this.randomImage} random={this.state.random} eraserChanged={this.eraserChanged} 
                  brushSize={this.state.brushSize} image={this.state.image} model={this.state.model}/>
            </Item>
            <Item sx={{width: '100%', alignSelf: 'center'}}> 
                  <Modified imageRandomed={this.imageRandomed} imageProcessed={this.imageProcessed}  
                  image={this.state.image} brushSize={this.state.brushSize} reset={this.state.reset} 
                  random={this.state.random}  ref={(c) => this.mod = c} model={this.state.model}
                  eraserEnable={this.state.eraserEnable}/>
            </Item>
            <Item sx={{flexShrink: 1}}> 
              <Help/>
            </Item>
         </Box>
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
