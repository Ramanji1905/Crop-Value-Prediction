import React from 'react';
import { View, Image, StyleSheet } from 'react-native';
const Loader = () => {
  return (
    <View style={styles.loader}>
      <Image source={require('./farmer.gif')} style={styles.loaderImage} />
    </View>
  );
};
const styles = StyleSheet.create({
  loader: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loaderImage: {
    width: 500, 
    height: 600,
  },
});
export default Loader;
