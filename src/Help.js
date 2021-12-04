import React, { Component } from 'react';
import {Card, CardHeader, CardContent, CardActions, Button} from '@material-ui/core';
import NavigationArrowForward from 'material-ui/svg-icons/navigation/arrow-forward';
import NavigationArrowBack from 'material-ui/svg-icons/navigation/arrow-back';
import {indigo500} from 'material-ui/styles/colors';
import Box from '@material-ui/core/Box';
import './App.css';
const styles = {
  button: {
    margin: 12,
  },
  exampleImageInput: {
    cursor: 'pointer',
    position: 'absolute',
    top: 0,
    bottom: 0,
    right: 0,
    left: 0,
    width: '100%',
    opacity: 0,
  },
};
function Item(props) {
  const { sx, ...other } = props;
  return (
    <Box
      sx={{
        // bgcolor: 'primary.main',
        // color: 'white',
        p: 1,
        m: 1,
        borderRadius: 1,
        // textAlign: 'center',
        fontSize: '1rem',
        fontWeight: '700',
        ...sx,
      }}
      {...other}
    />
  );
}

class Help extends Component {
  constructor(props) {
    super(props);
    
    this.state = {
      tSlide: 0,
      slides: [
        <span><b>交互式图像修复</b> 帮助你利用机器学习的力量“智能”修复图像.</span>,
        <span>通过选择待修复的图像类型与不同模型，模型可以对于随机或者大面积的缺失产生良好的效果</span>,
      ]
    };
  }

  nextPage = () => {
    if (this.state.tSlide !== this.state.slides.length - 1) {
      this.setState({
        tSlide: this.state.tSlide + 1
      });
    }
  }

  prevPage = () => {
    if (this.state.tSlide !== 0) {
      this.setState({
        tSlide: this.state.tSlide - 1
      });
    }
  }
  
  handleImageUpload(e) {
    this.props.imageUpload(e)
  }
  render() {
    return (
      <div className="box" id="help">
        <Box
            sx={{
            // textAlign: 'center',
            display: 'flex',
            justifyContent: 'space-around',
            // alignItems: 'center',
            p: 2,
            m: 2,
            // bgcolor: 'background.paper',
            }}
        >
          <Card >
            <CardHeader title="一点小帮助" titleColor={indigo500} titleStyle={{fontWeight: 800}}  />
            <CardContent >
              {this.state.slides[this.state.tSlide]}
            </CardContent>
            <CardActions>
              <Button startIcon={<NavigationArrowBack />} onClick={this.prevPage} />
              <Button endIcon={<NavigationArrowForward />} onClick={this.nextPage}/>
            </CardActions>
          </Card>
      </Box>
      </div>
      
    );
    
        
    
  }
}

export default Help;
