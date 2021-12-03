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
        <span><b>交互式人脸修复</b> allows you to explore how computers see by modifying images.</span>,
        <span>The <b>Class</b> column tells you what the computer thinks the image is, and the <b>Confidence %</b> column tells you how confident it is in its choice.</span>,
        <span>You can click on a row to see the <b>Class Activation Map</b>. This is a heatmap showing which areas of the image the computer found most important when choosing that class.</span>,
        <span>Hover over the <b>Modified Image</b> to see a yellow circle. Draw, by clicking and dragging over the image, to remove an object.</span>,
        <span>The <b>Absolute % Change</b> column shows you the difference between the original classication and the modified classification. Clicking on <b>Confidence %</b> sorts by the new top classes. You can also see the new <b>Class Activation Maps</b> for the modified image by clicking on a row.</span>,
        <span>Try out different images and see how the computer does! You can try taking the ball out of a soccer match, removing the poles from a skier, taking people out of a beach scene, and much more!</span>
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
