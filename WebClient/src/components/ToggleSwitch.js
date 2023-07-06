import React, {Component} from "react";
import "./ToggleSwitch.css";
  
class ToggleSwitch extends Component {
   render() {
     return (
       <div className="toggle-switch small-switch">
         <input
           type="checkbox"
          className="toggle-switch-checkbox"
          name="toggleSwitch"
           id="toggleSwitch"
           
        />
        <label className="toggle-switch-label" htmlFor="toggleSwitch">
          <span className="toggle-switch-inner" />
          <span className="toggle-switch-switch" />
        </label>
      </div>
    );
  }
}
  
export default ToggleSwitch;