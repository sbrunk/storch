import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Build deep learning models in Scala',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
          Storch provides GPU accelerated tensor operations, automatic differentiation,
          and a neural network API for building and training machine learning models.
      </>
    ),
  },
  {
    title: 'Get the best out of PyTorch & Scala',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Storch aims to be close to the original PyTorch API, while still leveraging Scala's powerful type
        system for safer tensor operations.
      </>
    ),
  },
  {
    title: 'Powered by LibTorch & JavaCPP',
    Svg: require('@site/static/img/PyTorch_logo_icon.svg').default,
    description: (
      <>
        <p>Storch is based on <a href="https://pytorch.org/cppdocs/">LibTorch</a> the C++ library underlying PyTorch.</p>

        <p><a href="https://github.com/bytedeco/javacpp">JavaCPP</a> provides us with generated JVM bindings and seamless,
        multiplatform interop with native code, including CUDA support for GPU acceleration.</p>
      </>
    ),
  },
];

function Feature({Svg, title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
